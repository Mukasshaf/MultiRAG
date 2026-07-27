
const chatMessages   = document.getElementById('chatMessages');
const welcomeScreen  = document.getElementById('welcomeScreen');
const queryInput     = document.getElementById('queryInput');
const sendBtn        = document.getElementById('sendBtn');
const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill   = document.getElementById('progressFill');
const progressText   = document.getElementById('progressText');
const docList        = document.getElementById('docList');
const emptyDocs      = document.getElementById('emptyDocs');
const docCount       = document.getElementById('docCount');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const btnHealth      = document.getElementById('btnHealth');
const toastContainer = document.getElementById('toastContainer');

let isStreaming = false;


async function checkHealth() {
  statusDot.className = 'status-dot';
  statusText.textContent = 'Checking...';
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const ollamaOk = data.ollama === 'ok';
    const pineconeOk = data.pinecone === 'ok';
    const activeModel = data.active_model || 'unknown';

    // Update UI elements for model
    const welcomeModelEl = document.getElementById('welcomeModelName');
    if (welcomeModelEl) welcomeModelEl.textContent = activeModel;

    const activeModelTag = document.getElementById('activeModelTag');
    if (activeModelTag) {
      activeModelTag.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg> ${activeModel} · local`;
    }

    if (ollamaOk && pineconeOk) {
      statusDot.className = 'status-dot ok';
      statusText.textContent = 'All systems ready';
    } else {
      statusDot.className = 'status-dot warn';
      const issues = [];
      if (!ollamaOk) issues.push(`Ollama: ${data.ollama}`);
      if (!pineconeOk) issues.push(`Pinecone: ${data.pinecone}`);
      statusText.textContent = issues.join(' · ');
      toast(issues.join('. '), 'error');
    }
  } catch (e) {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Cannot reach server';
    toast('Cannot reach the server. Is it running?', 'error');
  }
}
btnHealth.addEventListener('click', checkHealth);




async function refreshDocList() {
  try {
    const res = await fetch('/documents');
    const data = await res.json();
    const docs = data.documents || [];
    docCount.textContent = docs.length;

    if (docs.length === 0) {
      emptyDocs.classList.remove('hidden');
      // Remove any existing doc cards
      docList.querySelectorAll('.doc-card').forEach(el => el.remove());
      return;
    }
    emptyDocs.classList.add('hidden');

    // Rebuild list
    docList.querySelectorAll('.doc-card').forEach(el => el.remove());
    docs.forEach(doc => docList.appendChild(buildDocCard(doc)));
  } catch (e) {
    console.error('Failed to refresh doc list', e);
  }
}

function buildDocCard(doc) {
  const ext = doc.file_type || 'unknown';
  const sizeStr = formatBytes(doc.size_bytes || 0);

  const card = document.createElement('div');
  card.className = 'doc-card';
  card.dataset.filename = doc.filename;

  card.innerHTML = `
    <div class="doc-type-badge doc-type-${ext}">${ext}</div>
    <div class="doc-info">
      <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
      <div class="doc-meta">
        <span class="doc-size">${sizeStr}</span>
      </div>
    </div>
    <button class="doc-delete-btn" title="Remove document" data-filename="${doc.filename}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
        <path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
      </svg>
    </button>`;

  card.querySelector('.doc-delete-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    deleteDocument(doc.filename);
  });
  return card;
}

async function deleteDocument(filename) {
  try {
    const res = await fetch(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    if (res.ok) {
      toast(`Removed: ${filename}`, 'success');
      refreshDocList();
    } else {
      toast(`Failed to delete ${filename}`, 'error');
    }
  } catch (e) {
    toast(`Error: ${e.message}`, 'error');
  }
}


uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

async function handleFiles(files) {
  if (!files || files.length === 0) return;

  const formData = new FormData();
  for (const f of files) formData.append('files', f);

  
  uploadProgress.classList.remove('hidden');
  progressFill.style.width = '0%';
  progressText.textContent = `Uploading ${files.length} file(s)...`;


  let fakeProgress = 0;
  const ticker = setInterval(() => {
    fakeProgress = Math.min(fakeProgress + 3, 85);
    progressFill.style.width = fakeProgress + '%';
  }, 150);

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    clearInterval(ticker);
    progressFill.style.width = '100%';

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    const success = data.results.filter(r => r.status === 'success').length;
    const skipped = data.results.filter(r => r.status === 'skipped').length;
    const errors  = data.results.filter(r => r.status?.startsWith('error') || r.status === 'rejected' || r.status?.endsWith('error')).length;

    progressText.textContent = `Done: ${success} ingested, ${skipped} skipped, ${errors} errors`;

    if (success > 0) toast(`${success} document(s) ingested into Pinecone ✓`, 'success');
    if (errors > 0) toast(`${errors} file(s) failed to process`, 'error');

    refreshDocList();
  } catch (e) {
    clearInterval(ticker);
    progressText.textContent = `Upload failed: ${e.message}`;
    toast(`Upload error: ${e.message}`, 'error');
  } finally {
    setTimeout(() => {
      uploadProgress.classList.add('hidden');
      progressFill.style.width = '0%';
    }, 3000);
    fileInput.value = '';
  }
}



queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
});

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener('click', sendMessage);

function setQuery(text) {
  queryInput.value = text;
  queryInput.focus();
  queryInput.dispatchEvent(new Event('input'));
}
window.setQuery = setQuery;

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query || isStreaming) return;

  welcomeScreen?.remove();

  isStreaming = true;
  sendBtn.disabled = true;
  queryInput.value = '';
  queryInput.style.height = 'auto';

  appendMessage('user', query);

  const assistantMsg = appendMessage('assistant', null);
  const bubble = assistantMsg.querySelector('.message-bubble');
  bubble.innerHTML = `<div class="typing-indicator">
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  </div>`;

  let fullText = '';
  let sources = [];

  try {
    const res = await fetch('/chat/full', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 5 }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    fullText = data.answer || '';
    sources = data.sources || [];
  } catch (e) {
    fullText = `⚠️ Error: ${e.message}`;
    toast(`Chat error: ${e.message}`, 'error');
  }

  await typewriterRender(bubble, fullText);

  if (sources.length > 0) {
    appendSources(assistantMsg, sources);
  }

  isStreaming = false;
  sendBtn.disabled = false;
  scrollToBottom();
}

function typewriterRender(element, text) {
  return new Promise(resolve => {
    element.textContent = '';
    let i = 0;
    const speed = Math.max(5, Math.min(20, Math.floor(3000 / text.length)));
    const interval = setInterval(() => {
      element.textContent += text[i++];
      scrollToBottom();
      if (i >= text.length) {
        clearInterval(interval);
        // Re-render as proper HTML (preserving line breaks)
        element.innerHTML = markdownToHtml(text);
        resolve();
      }
    }, speed);
  });
}

function markdownToHtml(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
    .replace(/^/, '<p>').replace(/$/, '</p>');
}

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatarIcon = role === 'user'
    ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8V4H8"/><rect x="4" y="8" width="16" height="12" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>`;

  div.innerHTML = `
    <div class="message-avatar">${avatarIcon}</div>
    <div class="message-body">
      <div class="message-bubble">${text ? markdownToHtml(text) : ''}</div>
    </div>`;

  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function appendSources(messageEl, sources) {
  const body = messageEl.querySelector('.message-body');
  const section = document.createElement('div');
  section.className = 'sources-section';

  const toggle = document.createElement('button');
  toggle.className = 'sources-toggle';
  toggle.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
    Sources
    <span class="src-count">${sources.length}</span>`;
  section.appendChild(toggle);

  const list = document.createElement('div');
  list.className = 'sources-list hidden';

  sources.forEach((src, i) => {
    const scorePercent = Math.round(src.score * 100);
    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-score">${scorePercent}%</div>
      <div class="source-meta">
        <div class="source-file">${src.source_file}</div>
        <div class="source-tags">
          <span class="source-tag">Page ${src.page + 1}</span>
          <span class="source-tag lang">${src.language?.toUpperCase() || 'UNK'}</span>
          <span class="source-tag">${src.file_type?.toUpperCase()}</span>
          <span class="source-tag">Chunk #${src.chunk_index}</span>
        </div>
        <div class="source-snippet">${truncate(src.text, 140)}</div>
      </div>`;
    list.appendChild(card);
  });

  toggle.addEventListener('click', () => {
    const open = list.classList.toggle('hidden');
    toggle.classList.toggle('open', !list.classList.contains('hidden'));
  });

  section.appendChild(list);
  body.appendChild(section);
  scrollToBottom();
}


function toast(message, type = 'info', duration = 4000) {
  const icons = {
    success: `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
    error:   `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
    info:    `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `${icons[type] || icons.info}<span>${message}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(20px)';
    el.style.transition = '0.3s ease';
    setTimeout(() => el.remove(), 300);
  }, duration);
}


function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
}

function truncate(str, max) {
  if (!str) return '';
  return str.length <= max ? str : str.slice(0, max) + '…';
}


(async function init() {
  await checkHealth();
  await refreshDocList();
})();
