#!/usr/bin/env python3
"""
Video STT Subtitle Server — video upload, transcription, keyword highlight, subtitle overlay.
Port: 5055 (override with PORT env var).
"""
import os, sys, asyncio, json, re, uuid, threading, math, sqlite3, shutil, urllib.request, html
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from flask import Flask, request, jsonify, Response, render_template_string, send_file

# ── Config ───────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "5055"))
JOB_DIR = Path("/tmp/video_stt_jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("/tmp/video_stt_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Transcription engines
USE_WHISPER = True
USE_SENSEVOICE = False  # optional; heavy download on first use

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "deepseek/deepseek-chat"

# Cantonese AI (optional fallback)
CANTONESE_AI_KEY = os.environ.get("CANTONESE_AI_KEY", "")

# Azure (optional TTS — not primary here)
AZURE_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastasia")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB uploads

# ── DB ───────────────────────────────────────────────────────────────────
DB_PATH = str(JOB_DIR / "jobs.db")
con = sqlite3.connect(DB_PATH, check_same_thread=False)
con.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        status TEXT DEFAULT 'pending',
        engine TEXT,
        lang TEXT,
        srt_path TEXT,
        keywords_path TEXT,
        error TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")
con.commit()

# ── Transcription model (lazy load) ────────────────────────────────────
_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("[INIT] Loading Whisper base model…")
        _whisper_model = whisper.load_model("base")
        print("[INIT] Whisper ready.")
    return _whisper_model

# ── Helpers ─────────────────────────────────────────────────────────────

def _ffmpeg():
    """Return working ffmpeg path."""
    for p in ["/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg", "ffmpeg"]:
        if shutil.which(p):
            return p
    return "ffmpeg"


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video using ffmpeg."""
    import subprocess
    cmd = [_ffmpeg(), "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame",
           "-q:a", "2", audio_path]
    subprocess.run(cmd, check=True, capture_output=True)


def _secs_to_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def segments_to_srt(segments: list[dict]) -> str:
    """Convert Whisper-style segments to SRT."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0)
        end = seg.get("end", start + 2)
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{i}\n{_secs_to_srt_time(start)} --> {_secs_to_srt_time(end)}\n{text}\n")
    return "\n".join(lines)


# ── Keyword extraction (two-stage OpenRouter) ────────────────────────────

def _call_openrouter(messages: list, max_tokens: int = 300, temperature: float = 0):
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                 "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


_STOPWORDS = {"的","了","是","在","我","有","和","就","不","人","都","一","一个","上","也","很",
              "到","说","要","去","你","会","着","没有","看","好","自己","这","那","他","她","它",
              "们","個","係","咁","咗","呢","喺","得","而","之","與","及","或","但","嗰"}


def _extract_keywords(text: str, top_n: int = 25) -> list[str]:
    """Two-stage LLM keyword extraction for Cantonese/Chinese."""
    # Stage 1: sample up to 120 segments for topic words
    sentences = re.split(r'[。！？\.\!\?，,]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    step = max(1, len(sentences) // 120) if len(sentences) > 120 else 1
    sample = sentences[::step][:120]
    sample_text = "\n".join(sample)

    topic_prompt = f"""你係廣東話語音分析助手。以下係一段語音嘅部分字幕，請提取最多25個關鍵詞。

規則：
- 只揀名詞、專有名詞、品牌名、地名、術語
- 唔好揀單字（至少2個字）
- 唔好揀虛詞（係、咁、呢、喺）
- 用粵語口語風格

格式：每行一個詞，純文字。

字幕：
{sample_text}"""

    try:
        raw = _call_openrouter([{"role":"user","content":topic_prompt}], max_tokens=400)
        topics = [line.strip() for line in raw.split("\n") if line.strip() and len(line.strip()) > 1]
        topics = [t for t in topics if t not in _STOPWORDS]
    except Exception:
        topics = []

    # Stage 2: chunk-based extraction with topic hints
    all_kws = []
    chunk_size = 30
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        chunk_text = "\n".join(chunk)
        hint = "相關主題：" + "、".join(topics[:10]) if topics else ""
        prompt = f"""{hint}

以下係一段字幕，請提取關鍵詞（名詞/專有名詞）。
格式：詞語|詞語|詞語（用 | 分隔，唔好解釋）

{chunk_text}"""
        try:
            raw2 = _call_openrouter([{"role":"user","content":prompt}], max_tokens=300)
            kws = [w.strip() for w in raw2.split("|") if len(w.strip()) > 1]
            kws = [w for w in kws if w not in _STOPWORDS]
            all_kws.extend(kws)
        except Exception:
            pass

    # TF-IDF fallback if LLM coverage low
    if len(all_kws) < len(sentences) * 0.3:
        try:
            import jieba.analyse
            jieba_kws = jieba.analyse.extract_tags(text, topK=top_n)
            all_kws.extend(jieba_kws)
        except Exception:
            pass

    # Deduplicate and rank
    freq = {}
    for w in all_kws:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in ranked[:top_n]]


# ── Job processing ──────────────────────────────────────────────────────

def _set_status(job_id: str, status: str, **kwargs):
    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        c.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
        for k, v in kwargs.items():
            c.execute(f"UPDATE jobs SET {k}=? WHERE id=?", (v, job_id))
        c.commit()


def _update_job_sse(job_id: str, event_type: str, data: dict):
    """Push SSE event to job subscribers."""
    # Simple in-memory queue
    q = _sse_queues.get(job_id)
    if q is not None:
        q.put((event_type, data))


from queue import Queue
_sse_queues: dict[str, Queue] = {}


def process_job(job_id: str, video_path: str, engine: str, lang: str):
    try:
        _set_status(job_id, "extracting_audio")
        _update_job_sse(job_id, "status", {"text": "提取音頻中…"})

        audio_path = str(AUDIO_DIR / f"{job_id}.mp3")
        extract_audio(video_path, audio_path)

        _set_status(job_id, "transcribing")
        _update_job_sse(job_id, "status", {"text": "轉錄中…"})

        # Transcribe
        if engine == "whisper":
            model = get_whisper()
            whisper_lang = "zh" if lang == "yue" else (lang or None)
            result = model.transcribe(audio_path, language=whisper_lang)
            segments = result.get("segments", [])
        elif engine == "sensevoice":
            # Placeholder — SenseVoice integration would go here
            segments = []
        elif engine == "cantoneseai":
            segments = _cantoneseai_segments(audio_path, lang)
        else:
            model = get_whisper()
            whisper_lang = "zh" if lang == "yue" else (lang or None)
            result = model.transcribe(audio_path, language=whisper_lang)
            segments = result.get("segments", [])

        if not segments:
            raise RuntimeError("No transcription segments returned")

        # Fix zero-timestamps (SenseVoice sometimes returns all 0)
        if all(seg.get("start", 0) == 0 and seg.get("end", 0) == 0 for seg in segments):
            import soundfile as sf
            info = sf.info(audio_path)
            total = info.duration
            n = len(segments)
            for i, seg in enumerate(segments):
                seg["start"] = i * (total / n)
                seg["end"] = (i + 1) * (total / n)

        # Write SRT
        srt_content = segments_to_srt(segments)
        srt_path = str(JOB_DIR / f"{job_id}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # Extract keywords
        full_text = " ".join(seg.get("text", "") for seg in segments)
        _set_status(job_id, "extracting_keywords")
        _update_job_sse(job_id, "status", {"text": "提取關鍵詞…"})
        keywords = _extract_keywords(full_text)
        kw_path = str(JOB_DIR / f"{job_id}_keywords.json")
        with open(kw_path, "w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False)

        _set_status(job_id, "done", srt_path=srt_path, keywords_path=kw_path)
        _update_job_sse(job_id, "done", {
            "srt": f"/file/{job_id}.srt",
            "keywords": f"/file/{job_id}_keywords.json",
        })

    except Exception as e:
        _set_status(job_id, "error", error=str(e))
        _update_job_sse(job_id, "error", {"error": str(e)})


def _cantoneseai_segments(audio_path: str, lang: str) -> list[dict]:
    """Cantonese.ai transcription — OpenAI-compatible endpoint."""
    import urllib.request, mimetypes
    if not CANTONESE_AI_KEY:
        raise RuntimeError("No CANTONESE_AI_KEY set")

    boundary = "----CantoneseAI"
    fname = os.path.basename(audio_path)
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "audio/mpeg"

    with open(audio_path, "rb") as f:
        file_data = f.read()

    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
            f"Content-Type: {mime}\r\n\r\n").encode() + file_data + b"\r\n"
    body += (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="model"\r\n\r\n'
             f"cantonese-1\r\n").encode()
    body += (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
             f"verbose_json\r\n").encode()
    body += (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="timestamp_granularities[]"\r\n\r\n'
             f"segment\r\n").encode()
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "https://api.cantonese.ai/v1/audio/transcriptions",
        data=body,
        headers={"Authorization": f"Bearer {CANTONESE_AI_KEY}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("segments", [])


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "No video file"}), 400
    engine = request.form.get("engine", "whisper")
    lang = request.form.get("lang", "yue")

    job_id = uuid.uuid4().hex
    vid_path = str(JOB_DIR / f"{job_id}_video")
    f.save(vid_path)

    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        c.execute("INSERT INTO jobs (id, status, engine, lang) VALUES (?, 'pending', ?, ?)",
                   (job_id, engine, lang))
        c.commit()

    threading.Thread(target=process_job, args=(job_id, vid_path, engine, lang), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/events/<job_id>")
def events(job_id):
    def gen():
        q = Queue()
        _sse_queues[job_id] = q
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                event_type, data = q.get(timeout=300)
                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if event_type in ("done", "error"):
                    break
        except Exception:
            pass
        finally:
            _sse_queues.pop(job_id, None)
    return Response(gen(), mimetype="text/event-stream")


@app.route("/status/<job_id>")
def status(job_id):
    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        row = c.execute("SELECT status, error, srt_path, keywords_path FROM jobs WHERE id=?",
                        (job_id,)).fetchone()
    if not row:
        return jsonify({"error": "Unknown job"}), 404
    status, error, srt, kw = row
    return jsonify({"status": status, "error": error,
                    "srt": f"/file/{job_id}.srt" if srt else None,
                    "keywords": f"/file/{job_id}_keywords.json" if kw else None})


@app.route("/file/<job_id>.srt")
def serve_srt(job_id):
    p = JOB_DIR / f"{job_id}.srt"
    if not p.exists():
        return "Not found", 404
    return Response(p.read_text(encoding="utf-8"), mimetype="text/plain")


@app.route("/file/<job_id>_keywords.json")
def serve_keywords(job_id):
    p = JOB_DIR / f"{job_id}_keywords.json"
    if not p.exists():
        return jsonify([])
    return Response(p.read_text(encoding="utf-8"), mimetype="application/json")


FFMPEG_FULL = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"


@app.route("/burn/<job_id>")
def burn_video(job_id):
    """Burn SRT subtitles into the original video using ffmpeg-full."""
    srt_path = JOB_DIR / f"{job_id}.srt"
    vid_path = JOB_DIR / f"{job_id}_video"
    out_path = JOB_DIR / f"{job_id}_subtitled.mp4"

    if not srt_path.exists():
        return jsonify({"error": "SRT not found — transcription incomplete"}), 404
    if not vid_path.exists():
        return jsonify({"error": "Original video not found"}), 404

    import subprocess
    cmd = [
        FFMPEG_FULL, "-y", "-i", str(vid_path),
        "-vf", f"subtitles={str(srt_path)}:force_style='FontName=PingFang HK,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Alignment=2'",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return send_file(str(out_path), mimetype="video/mp4", as_attachment=True,
                         download_name=f"{job_id}_subtitled.mp4")
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"ffmpeg failed: {e.stderr.decode()[:200]}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/file/<job_id>_subtitled.mp4")
def serve_burned(job_id):
    p = JOB_DIR / f"{job_id}_subtitled.mp4"
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(p), mimetype="video/mp4")


# ── HTML Frontend ──────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Video STT — 字幕提取 + 關鍵詞高亮</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#0f0f14;color:#e8e8e8;min-height:100vh}
  .container{max-width:900px;margin:0 auto;padding:32px 20px}
  h1{font-size:1.6rem;margin-bottom:8px}
  .subtitle{color:#888;font-size:0.9rem;margin-bottom:24px}
  .card{background:#1a1a24;border-radius:16px;padding:24px;margin-bottom:20px;border:1px solid #2a2a38}
  select, input[type=file]{width:100%;background:#0f0f14;border:1px solid #2a2a38;border-radius:10px;color:#e8e8e8;padding:12px;font-size:1rem;outline:none}
  select:focus{border-color:#a78bfa}
  .row{display:flex;gap:12px;margin-top:16px}
  button{border:none;border-radius:10px;padding:12px 20px;font-size:1rem;cursor:pointer;font-weight:600}
  .btn-primary{background:linear-gradient(135deg,#c9a227,#a78bfa);color:#0f0f14}
  .btn-secondary{background:#2a2a38;color:#aaa}
  .status{margin-top:14px;font-size:0.9rem;min-height:24px}
  .status.ok{color:#4ade80}
  .status.err{color:#f87171}
  .status.loading{color:#fbbf24}
  #video-container{position:relative;margin-top:16px;display:none}
  video{width:100%;border-radius:12px;background:#000}
  .cue-layer{position:absolute;bottom:60px;left:0;right:0;text-align:center;padding:0 20px;pointer-events:none}
  .cue-layer span{background:rgba(0,0,0,0.7);color:#fff;padding:6px 14px;border-radius:8px;font-size:1.1rem;line-height:1.6}
  .cue-layer span mark{background:transparent;color:#ff4444;font-weight:bold}
  #keywords{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
  #keywords span{background:#2a2a38;color:#c9a227;padding:6px 12px;border-radius:20px;font-size:0.85rem}
  .progress-bar{height:4px;background:#2a2a38;border-radius:2px;margin-top:12px;overflow:hidden;display:none}
  .progress-fill{height:100%;background:linear-gradient(90deg,#c9a227,#a78bfa);width:0%;transition:width .3s}
</style>
</head>
<body>
<div class="container">
  <h1>🎬 Video STT — 字幕提取 + 關鍵詞高亮</h1>
  <p class="subtitle">上傳影片 → 自動轉錄 → AI 關鍵詞 → 字幕覆蓋</p>

  <div class="card">
    <div style="display:flex;gap:12px;margin-bottom:12px">
      <select id="engine">
        <option value="whisper" selected>Whisper（本地，免費）</option>
        <option value="cantoneseai">Cantonese AI（廣東話專用）</option>
      </select>
      <select id="lang">
        <option value="yue" selected>粵語 Cantonese</option>
        <option value="zh">普通話 Mandarin</option>
        <option value="en">English</option>
      </select>
    </div>
    <input type="file" id="video-file" accept="video/*">
    <div class="progress-bar" id="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <div class="row">
      <button class="btn-primary" onclick="doUpload()">🎬 上傳並轉錄</button>
      <button class="btn-secondary" onclick="clearAll()">清除</button>
    </div>
    <div class="status" id="status"></div>
  </div>

  <div class="card" id="result-card" style="display:none">
    <div id="video-container">
      <video id="player" controls></video>
      <div class="cue-layer" id="cue"><span></span></div>
    </div>
    <div id="keywords"></div>
    <div style="margin-top:16px;display:flex;gap:12px">
      <a id="srt-link" class="btn-secondary" style="display:inline-block;text-decoration:none" download>⬇ 下載 SRT</a>
      <button class="btn-secondary" id="burn-btn" onclick="doBurn()">🔥 嵌入字幕並下載 MP4</button>
    </div>
  </div>
</div>

<script>
let cues = [], keywords = [], es = null;

function setStatus(msg, type='ok'){
  const el = document.getElementById('status');
  el.textContent = msg; el.className = 'status ' + type;
}

function clearAll(){
  document.getElementById('video-file').value = '';
  document.getElementById('result-card').style.display = 'none';
  document.getElementById('video-container').style.display = 'none';
  if(es){ es.close(); es=null; }
  cues=[]; keywords=[];
}

async function doUpload(){
  const file = document.getElementById('video-file').files[0];
  if(!file){ setStatus('請選擇影片','err'); return; }
  setStatus('⏳ 上傳中…','loading');
  document.querySelector('.btn-primary').disabled = true;
  document.getElementById('progress-bar').style.display = 'block';

  const fd = new FormData();
  fd.append('video', file);
  fd.append('engine', document.getElementById('engine').value);
  fd.append('lang', document.getElementById('lang').value);

  try {
    const r = await fetch('/upload', {method:'POST', body:fd});
    const d = await r.json();
    if(d.error) throw new Error(d.error);
    pollJob(d.job_id, file);
  } catch(e){
    setStatus('❌ '+e.message,'err');
    document.querySelector('.btn-primary').disabled = false;
  }
}

function pollJob(jobId, file){
  setStatus('⏳ 處理中，請稍候…','loading');
  // SSE for real-time updates
  es = new EventSource('/events/'+jobId);
  es.addEventListener('status', e => {
    const d = JSON.parse(e.data);
    setStatus(d.text, 'loading');
  });
  es.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    showResult(file, d.srt, d.keywords, jobId);
    es.close();
  });
  es.addEventListener('error', e => {
    const d = JSON.parse(e.data);
    setStatus('❌ '+d.error, 'err');
    es.close();
    document.querySelector('.btn-primary').disabled = false;
  });
  // Fallback polling if SSE drops
  es.onerror = () => {
    es.close();
    fallbackPoll(jobId, file);
  };
}

async function fallbackPoll(jobId, file){
  for(let i=0;i<300;i++){
    await new Promise(r=>setTimeout(r,2000));
    const r = await fetch('/status/'+jobId);
    const d = await r.json();
    if(d.status==='done'){ showResult(file, d.srt, d.keywords, jobId); return; }
    if(d.status==='error'){ setStatus('❌ '+d.error,'err'); document.querySelector('.btn-primary').disabled=false; return; }
    setStatus(`⏳ 處理中… ${d.status}`,'loading');
  }
  setStatus('⏳ 超時，請檢查狀態','err');
  document.querySelector('.btn-primary').disabled = false;
}

async function showResult(file, srtUrl, kwUrl, jobId){
  currentJobId = jobId;
  setStatus('✅ 完成！','ok');
  document.querySelector('.btn-primary').disabled = false;
  document.getElementById('progress-bar').style.display = 'none';

  // Load video
  const video = document.getElementById('player');
  video.src = URL.createObjectURL(file);
  document.getElementById('video-container').style.display = 'block';
  document.getElementById('result-card').style.display = 'block';

  // Load keywords
  try {
    const kr = await fetch(kwUrl);
    keywords = await kr.json();
    const kwDiv = document.getElementById('keywords');
    kwDiv.innerHTML = keywords.map(k=>`<span>${k}</span>`).join('');
  } catch(e){ keywords=[]; }

  // Load SRT
  try {
    const sr = await fetch(srtUrl);
    const srtText = await sr.text();
    cues = parseSRT(srtText);
    attachSubtitle(video);
  } catch(e){ setStatus('SRT 載入失敗: '+e.message,'err'); }

  document.getElementById('srt-link').href = srtUrl;
  document.getElementById('srt-link').download = jobId + '.srt';
}

function parseSRT(text){
  const lines = text.trim().split(/\r?\n/);
  const out = [];
  let i=0;
  while(i<lines.length){
    if(!lines[i].trim() || lines[i].trim().match(/^\d+$/)){
      if(lines[i].trim().match(/^\d+$/)) i++;
      if(i>=lines.length) break;
      const timeLine = lines[i++];
      const match = timeLine.match(/([\d:,]+)\s*-->\s*([\d:,]+)/);
      if(!match){ i++; continue; }
      const start = srtTimeToSecs(match[1]);
      const end = srtTimeToSecs(match[2]);
      let textLines = [];
      while(i<lines.length && lines[i].trim()){ textLines.push(lines[i].trim()); i++; }
      out.push({start, end, text: textLines.join('\n')});
    } else { i++; }
  }
  return out;
}

function srtTimeToSecs(t){
  const m = t.match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
  if(!m) return 0;
  return parseInt(m[1])*3600 + parseInt(m[2])*60 + parseInt(m[3]) + parseInt(m[4])/1000;
}

function highlightText(text){
  if(!keywords.length) return text;
  const sorted = [...keywords].sort((a,b)=>b.length-a.length);
  let out = text;
  for(const kw of sorted){
    const re = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')})`,'g');
    out = out.replace(re,'<mark>$1</mark>');
  }
  return out;
}

function attachSubtitle(video){
  const cue = document.querySelector('#cue span');
  video.addEventListener('timeupdate', () => {
    const t = video.currentTime;
    const seg = cues.find(c => c.start <= t && t <= c.end);
    if(seg) cue.innerHTML = highlightText(seg.text);
    else cue.textContent = '';
  });
}

let currentJobId = '';

async function doBurn(){
  if(!currentJobId){ setStatus('請先上傳並轉錄影片','err'); return; }
  setStatus('⏳ 嵌入字幕中…','loading');
  document.getElementById('burn-btn').disabled = true;
  try {
    const r = await fetch('/burn/' + currentJobId, {method: 'GET'});
    if(r.ok){
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = currentJobId + '_subtitled.mp4';
      a.click();
      setStatus('✅ 字幕已嵌入並下載','ok');
    } else {
      const d = await r.json();
      throw new Error(d.error || 'Burn failed');
    }
  } catch(e){
    setStatus('❌ ' + e.message, 'err');
  }
  document.getElementById('burn-btn').disabled = false;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
