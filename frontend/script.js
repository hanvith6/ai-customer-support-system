const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const typingIndicator = document.getElementById("typingIndicator");
const chatContainer = document.getElementById("chatContainer");
const widgetToggle = document.getElementById("widgetToggle");
const toggleIcon = document.getElementById("toggleIcon");
const closeChat = document.getElementById("closeChat");
const quickReplies = document.getElementById("quickReplies");

const API_BASE = window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "";
const API_URL = API_BASE + "/chat";

// ── Widget toggle (integrated from AI-Chatbot-DL-NLP chatbox pattern) ─

function openWidget() {
    chatContainer.classList.add("open");
    widgetToggle.classList.add("active");
    toggleIcon.innerHTML = "&#x2715;";
    userInput.focus();
}

function closeWidget() {
    chatContainer.classList.remove("open");
    widgetToggle.classList.remove("active");
    toggleIcon.innerHTML = "&#x1f4ac;";
}

widgetToggle.addEventListener("click", function () {
    if (chatContainer.classList.contains("open")) {
        closeWidget();
    } else {
        openWidget();
    }
});

closeChat.addEventListener("click", function () {
    closeWidget();
});

// ── Helpers ────────────────────────────────────────────────────────────

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    typingIndicator.style.display = "flex";
    scrollToBottom();
}

function hideTyping() {
    typingIndicator.style.display = "none";
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ── Quick replies (integrated from AI-Chatbot-DL-NLP React UI) ───────

quickReplies.addEventListener("click", function (e) {
    const btn = e.target.closest(".quick-reply-btn");
    if (!btn) return;

    const msg = btn.getAttribute("data-msg");
    if (!msg) return;

    addUserMessage(msg);
    sendMessage(msg);

    // Hide quick replies after first use
    quickReplies.style.display = "none";
});

// ── Render messages ───────────────────────────────────────────────────

function addUserMessage(text) {
    const el = document.createElement("div");
    el.className = "message user-message";
    el.innerHTML = `
        <div class="message-content">${escapeHtml(text)}</div>
        <div class="message-meta">You</div>
    `;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function addBotMessage(data) {
    const el = document.createElement("div");
    el.className = "message bot-message";

    let pills = "";

    // Intent pill
    if (data.intent && data.intent !== "unknown") {
        pills += `<span class="pill pill-intent">${escapeHtml(data.intent)}</span>`;
    }

    // Sentiment pill
    if (data.sentiment && data.sentiment.label) {
        const cls = "pill-sentiment-" + data.sentiment.label;
        pills += `<span class="pill ${cls}">${data.sentiment.label} ${(data.sentiment.score * 100).toFixed(0)}%</span>`;
    }

    // Entity pills
    if (data.entities && data.entities.length > 0) {
        data.entities.forEach(function (ent) {
            pills += `<span class="pill pill-entity">${escapeHtml(ent.word)} (${escapeHtml(ent.entity)})</span>`;
        });
    }

    const pillsHtml = pills ? `<div class="entity-pills">${pills}</div>` : "";

    el.innerHTML = `
        <div class="message-content">${escapeHtml(data.response)}</div>
        ${pillsHtml}
        <div class="message-meta">Bot</div>
    `;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function addErrorMessage(msg) {
    const el = document.createElement("div");
    el.className = "message bot-message";
    el.innerHTML = `
        <div class="message-content" style="color:#991b1b">${escapeHtml(msg)}</div>
        <div class="message-meta">Bot</div>
    `;
    chatMessages.appendChild(el);
    scrollToBottom();
}

// ── API call ──────────────────────────────────────────────────────────

async function sendMessage(text) {
    sendBtn.disabled = true;
    showTyping();

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text }),
        });

        hideTyping();

        if (!res.ok) {
            const err = await res.json().catch(function () {
                return { detail: "Server error" };
            });
            addErrorMessage(err.detail || "Something went wrong. Please try again.");
            return;
        }

        const data = await res.json();
        addBotMessage(data);
    } catch (e) {
        hideTyping();
        addErrorMessage("Unable to reach the server. Please check your connection.");
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// ── Form submit ───────────────────────────────────────────────────────

chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    addUserMessage(text);
    userInput.value = "";
    sendMessage(text);
});

// ── Auto-open widget on page load ─────────────────────────────────────

setTimeout(openWidget, 600);
