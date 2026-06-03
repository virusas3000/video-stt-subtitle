
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
  const lines = text.trim().replace(/\\r/g, "").split("\\n");
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
