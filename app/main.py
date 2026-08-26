from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .agent import SupportAgent

app = FastAPI(title="Aster & Row Support Agent")
agent = SupportAgent()

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aster & Row Support</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      height: 100%;
      background: #e5ddd5;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .app {
      width: 100%;
      max-width: 720px;
      height: 94vh;
      background: #efeae2;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 16px rgba(0,0,0,0.15);
      border-radius: 12px;
      overflow: hidden;
    }

    /* Header */
    .header {
      background: #1e3a2b;
      color: #ffffff;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .avatar {
      width: 38px;
      height: 38px;
      border-radius: 50%;
      background: #2a523d;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
      font-size: 17px;
    }
    .header-info { flex: 1; }
    .title { font-size: 16px; font-weight: 600; }
    .status { font-size: 12px; opacity: 0.85; }
    .btn-new {
      background: rgba(255,255,255,0.18);
      border: 0;
      color: #fff;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
    }
    .btn-new:hover { background: rgba(255,255,255,0.28); }

    /* Chat Body */
    .chat-body {
      flex: 1;
      overflow-y: auto;
      padding: 16px 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      -webkit-overflow-scrolling: touch;
    }

    .msg-wrap {
      display: flex;
      flex-direction: column;
      max-width: 85%;
    }
    .msg-wrap.user { align-self: flex-end; }
    .msg-wrap.agent { align-self: flex-start; }

    .bubble {
      padding: 10px 14px;
      font-size: 14.5px;
      line-height: 1.45;
      color: #111b21;
      white-space: pre-wrap;
      box-shadow: 0 1px 1px rgba(0,0,0,0.08);
    }

    .msg-wrap.user .bubble {
      background: #d9fdd3;
      border-radius: 10px 10px 0px 10px;
    }

    .msg-wrap.agent .bubble {
      background: #ffffff;
      border-radius: 10px 10px 10px 0px;
      border: 1px solid #e2e8f0;
    }

    /* Meta Tags (Category & Citation Proofs) */
    .meta-tags {
      margin-top: 5px;
      display: flex;
      gap: 6px;
      align-items: center;
      flex-wrap: wrap;
    }
    .tag-category {
      font-weight: 700;
      font-size: 10px;
      padding: 2px 7px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .tag-ANSWER { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .tag-CLARIFY { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
    .tag-ABSTAIN { background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
    .tag-HANDOFF { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .tag-REFUSE { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

    .tag-citation {
      background: #ffffff;
      color: #334155;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      padding: 2px 7px;
      font-size: 11px;
    }
    .tag-tool {
      background: #e0f2fe;
      color: #0369a1;
      border: 1px solid #bae6fd;
      border-radius: 4px;
      padding: 2px 7px;
      font-family: monospace;
      font-size: 11px;
    }

    /* Input Footer */
    .footer {
      background: #f0f2f5;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      border-top: 1px solid #e2e8f0;
    }
    .input-box {
      flex: 1;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 20px;
      padding: 10px 16px;
      font-size: 14.5px;
      color: #111b21;
      outline: none;
    }
    .input-box:focus { border-color: #1e3a2b; }

    .btn-send {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background: #1e3a2b;
      color: #ffffff;
      border: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      flex-shrink: 0;
    }
    .btn-send:hover { background: #15291e; }
    .btn-send:disabled { opacity: 0.5; cursor: wait; }
  </style>
</head>
<body>

  <div class="app">
    <header class="header">
      <div class="avatar">A</div>
      <div class="header-info">
        <div class="title">Aster &amp; Row Support</div>
        <div class="status">online</div>
      </div>
      <button class="btn-new" onclick="resetChat()">New Chat</button>
    </header>

    <div id="chatBody" class="chat-body">
      <div class="msg-wrap agent">
        <div class="bubble">Hi! Welcome to Aster & Row Support. How can I help you today?

Share your order ID (e.g. <b>ORD-1007</b>) for real-time tracking & delivery lookups.</div>
        <div class="meta-tags">
          <span class="tag-category tag-ANSWER">Category: ANSWER</span>
        </div>
      </div>
    </div>

    <form id="chatForm" class="footer">
      <input type="text" id="msgInput" class="input-box" placeholder="Type your message..." required autocomplete="off" />
      <button type="submit" id="sendBtn" class="btn-send">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
      </button>
    </form>
  </div>

  <script>
    const chatBody = document.getElementById('chatBody');
    const chatForm = document.getElementById('chatForm');
    const msgInput = document.getElementById('msgInput');
    const sendBtn = document.getElementById('sendBtn');

    let sid = localStorage.getItem('ar_sid') || crypto.randomUUID();
    localStorage.setItem('ar_sid', sid);

    function resetChat() {
      sid = crypto.randomUUID();
      localStorage.setItem('ar_sid', sid);
      chatBody.innerHTML = `
        <div class="msg-wrap agent">
          <div class="bubble">New session started. How can I assist you?</div>
          <div class="meta-tags"><span class="tag-category tag-ANSWER">Category: ANSWER</span></div>
        </div>
      `;
    }

    function appendMsg(text, role, data = null) {
      const wrap = document.createElement('div');
      wrap.className = `msg-wrap ${role}`;
      
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerHTML = text.replace(/\\n/g, '<br>');
      wrap.appendChild(bubble);

      if (data && role === 'agent') {
        const meta = document.createElement('div');
        meta.className = 'meta-tags';
        
        meta.innerHTML = `<span class="tag-category tag-${data.state}">Category: ${data.state}</span>`;

        if (data.citations && data.citations.length > 0) {
          data.citations.forEach(c => {
            meta.innerHTML += `<span class="tag-citation">Proof: ${c.filename} — ${c.heading}</span>`;
          });
        }

        if (data.tool_calls && data.tool_calls.length > 0) {
          data.tool_calls.forEach(t => {
            meta.innerHTML += `<span class="tag-tool">Tool: ${t.name}(${t.arguments.order_id || ''})</span>`;
          });
        }

        wrap.appendChild(meta);
      }

      chatBody.appendChild(wrap);
      chatBody.scrollTop = chatBody.scrollHeight;
      return wrap;
    }

    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = msgInput.value.trim();
      if (!text) return;

      appendMsg(text, 'user');
      msgInput.value = '';
      sendBtn.disabled = true;

      const loading = appendMsg('Checking policies...', 'agent');

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, session_id: sid })
        });
        const data = await res.json();
        chatBody.removeChild(loading);
        const cleanAnswer = data.answer.replace(/\\n\\nSources:[\\s\\S]*$/, '');
        appendMsg(cleanAnswer, 'agent', data);
      } catch (err) {
        chatBody.removeChild(loading);
        appendMsg('Unable to reach support service.', 'agent');
      } finally {
        sendBtn.disabled = false;
        msgInput.focus();
      }
    });

    msgInput.focus();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=PAGE, headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
def chat(request: ChatRequest):
    return agent.respond(request.message, request.session_id)

@app.get("/traces/{trace_id}")
def trace(trace_id: str):
    for item in agent.traces:
        if item["trace_id"] == trace_id:
            return item
    return {"error": "trace not found"}
