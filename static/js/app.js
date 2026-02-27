// ================================================================
//  MindScan — Frontend Application Logic
//  Medusmo-Inspired Dark Cinematic UI
//  Includes: Navigation, Scroll Animations, PHQ-8, Interview,
//            Webcam + Face Detection, Results Rendering
// ================================================================

// ── State ──────────────────────────────────────────────────────
let currentSection = 'landing';
let phqAnswers = Array(8).fill(-1);
let phqIndex = 0;
let interviewIndex = 0;
let interviewResponses = [];
let webcamStream = null;

// Face detection state
let faceModelsLoaded = false;
let faceDetectionInterval = null;
let expressionHistory = [];
let faceDetectedCount = 0;
let totalDetectionAttempts = 0;

// Audio state — TTS + STT
let isSpeaking = false;
let isRecording = false;
let recognition = null;
let audioContext = null;
let analyser = null;
let micStream = null;
let waveformAnimId = null;
let recTimerInterval = null;
let recSeconds = 0;

// ────────────────────────────────────────────────────────────────
//  INIT ON LOAD
// ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initWaveforms();
    initNavScroll();
    initCounters();
});

// ── Scroll-triggered animations ─────────────────────────────────
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Trigger children stagger
                if (entry.target.querySelector('.stagger')) {
                    const children = entry.target.querySelectorAll('.stagger > *');
                    children.forEach((child, i) => {
                        child.style.setProperty('--i', i);
                        setTimeout(() => child.classList.add('visible'), i * 100);
                    });
                }
            }
        });
    }, { threshold: 0.15 });

    document.querySelectorAll('.animate-up').forEach(el => observer.observe(el));
    // Also observe stagger containers directly
    document.querySelectorAll('.stagger').forEach(el => {
        const obs = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const children = entry.target.children;
                    Array.from(children).forEach((child, i) => {
                        setTimeout(() => child.classList.add('visible'), i * 120);
                    });
                }
            });
        }, { threshold: 0.1 });
        obs.observe(el);
    });
}

// ── Animated waveform bars ──────────────────────────────────────
function initWaveforms() {
    const colors = { 'wave-text': '#4F8EF7', 'wave-audio': '#9B6FFF', 'wave-visual': '#10B981' };
    Object.keys(colors).forEach(id => {
        const container = document.getElementById(id);
        if (!container) return;
        const barCount = 40;
        for (let i = 0; i < barCount; i++) {
            const bar = document.createElement('span');
            const h = 15 + Math.random() * 85;
            bar.style.height = h + '%';
            bar.style.background = colors[id];
            bar.style.animationDelay = (Math.random() * 1.5) + 's';
            bar.style.animationDuration = (1 + Math.random() * 1) + 's';
            container.appendChild(bar);
        }
    });
}

// ── Navbar scroll behavior ──────────────────────────────────────
function initNavScroll() {
    const nav = document.getElementById('main-nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });
}

// ── Counter animation (stats) ───────────────────────────────────
function initCounters() {
    const counters = document.querySelectorAll('.stat-number');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.dataset.animated) {
                entry.target.dataset.animated = 'true';
                const target = parseFloat(entry.target.dataset.target);
                const suffix = entry.target.dataset.suffix || '';
                const isFloat = target % 1 !== 0;
                const prefix = target >= 300 ? '' : '';  // No prefix needed
                const postfix = target >= 300 ? 'M+' : suffix;
                animateCounter(entry.target, target, postfix, isFloat);
            }
        });
    }, { threshold: 0.5 });
    counters.forEach(c => observer.observe(c));
}

function animateCounter(el, target, suffix, isFloat) {
    const duration = 1800;
    let start = null;
    const easeOut = t => 1 - Math.pow(1 - t, 3);

    const step = (ts) => {
        if (!start) start = ts;
        const progress = Math.min((ts - start) / duration, 1);
        const val = easeOut(progress) * target;
        if (isFloat) {
            el.textContent = val.toFixed(2) + suffix;
        } else {
            el.textContent = Math.floor(val) + suffix;
        }
        if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

// ── PHQ-8 Questions ────────────────────────────────────────────
const PHQ_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual"
];

const PHQ_OPTIONS = [
    { key: '0', label: 'Not at all', value: 0 },
    { key: '1', label: 'Several days', value: 1 },
    { key: '2', label: 'More than half the days', value: 2 },
    { key: '3', label: 'Nearly every day', value: 3 }
];

// ── Interview Questions ────────────────────────────────────────
const INTERVIEW_QUESTIONS = [
    "Hello! I'm **Mira**, your MindScan virtual assistant. I'd like to ask you a few questions to better understand how you've been feeling. There are no right or wrong answers — just share whatever feels comfortable.\n\nLet's start: **How have you been feeling lately?**",
    "Thank you for sharing that. **How has your sleep been recently?** Do you have trouble falling asleep, staying asleep, or do you find yourself sleeping too much?",
    "I appreciate you telling me. **How would you describe your energy levels day to day?** Do you feel tired often?",
    "**Have you lost interest or pleasure in activities you used to enjoy?** Things like hobbies, socializing, or work?",
    "I understand. **How do you feel about the future?** Do you generally feel hopeful or do things seem difficult?",
    "**Do you find it difficult to concentrate on tasks?** For example, reading, watching TV, or following conversations?",
    "**How are your relationships with people close to you?** Do you feel connected to others or more isolated?",
    "Thank you for being open with me. Last question: **Is there anything else that has been bothering you recently?** Anything you'd like to share?"
];


// ═══════════════════════════════════════════════════════════════════
//  NAVIGATION
// ═══════════════════════════════════════════════════════════════════
function goTo(sectionId) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');

    // Nav state
    const navLinks = document.getElementById('nav-links-main');
    const navSteps = document.getElementById('nav-steps');

    if (sectionId === 'landing') {
        navLinks.style.display = 'flex';
        navSteps.style.display = 'none';
    } else {
        navLinks.style.display = 'none';
        navSteps.style.display = 'flex';
    }

    // Step dots
    const steps = ['landing', 'phq', 'interview', 'results'];
    steps.forEach((s, i) => {
        const dot = document.getElementById('dot-' + s);
        dot.classList.remove('active', 'done');
        const targetIdx = steps.indexOf(sectionId);
        if (i < targetIdx) dot.classList.add('done');
        if (i === targetIdx) dot.classList.add('active');
    });

    currentSection = sectionId;
    window.scrollTo(0, 0);
    if (sectionId === 'phq') renderPhqQuestion();
    if (sectionId === 'interview') startInterview();
}


// ═══════════════════════════════════════════════════════════════════
//  PHQ-8 SURVEY
// ═══════════════════════════════════════════════════════════════════
function renderPhqQuestion() {
    const card = document.getElementById('phq-question-card');
    card.innerHTML = `
    <div class="question-number">Question ${phqIndex + 1} of 8</div>
    <div class="question-text">${PHQ_QUESTIONS[phqIndex]}</div>
    <div class="options-list">
      ${PHQ_OPTIONS.map(opt => `
        <button class="option-btn ${phqAnswers[phqIndex] === opt.value ? 'selected' : ''}"
                onclick="selectPhqOption(${opt.value})">
          <span class="option-key">${opt.key}</span>
          ${opt.label}
        </button>
      `).join('')}
    </div>`;

    const pct = ((phqIndex + 1) / 8) * 100;
    document.getElementById('phq-progress').style.width = pct + '%';
    document.getElementById('phq-progress-label').textContent = `${phqIndex + 1} / 8`;
    document.getElementById('phq-back-btn').disabled = phqIndex === 0;
    updatePhqNextBtn();
}

function selectPhqOption(value) {
    phqAnswers[phqIndex] = value;
    renderPhqQuestion();
}

function updatePhqNextBtn() {
    const btn = document.getElementById('phq-next-btn');
    btn.disabled = phqAnswers[phqIndex] < 0;
    btn.textContent = phqIndex === 7 ? 'Continue to Interview →' : 'Next →';
}

function phqNext() {
    if (phqAnswers[phqIndex] < 0) return;
    if (phqIndex < 7) { phqIndex++; renderPhqQuestion(); }
    else goTo('interview');
}

function phqBack() {
    if (phqIndex > 0) { phqIndex--; renderPhqQuestion(); }
}


// ═══════════════════════════════════════════════════════════════════
//  VIRTUAL ASSISTANT INTERVIEW
// ═══════════════════════════════════════════════════════════════════
function startInterview() {
    interviewIndex = 0;
    interviewResponses = [];
    document.getElementById('chat-messages').innerHTML = '';
    renderInterviewProgress();
    setTimeout(() => askQuestion(0), 600);
}

function renderInterviewProgress() {
    const list = document.getElementById('q-progress-list');
    const labels = ['General wellbeing', 'Sleep patterns', 'Energy levels',
        'Interest & pleasure', 'Future outlook', 'Concentration', 'Relationships', 'Other concerns'];
    list.innerHTML = labels.map((label, i) => {
        let cls = i < interviewIndex ? 'done' : i === interviewIndex ? 'current' : '';
        return `<div class="q-progress-item ${cls}">
      <span class="q-icon">${i < interviewIndex ? '✓' : i + 1}</span>${label}</div>`;
    }).join('');
}

function askQuestion(idx) {
    if (idx >= INTERVIEW_QUESTIONS.length) { finishInterview(); return; }
    const messages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;

    setTimeout(() => {
        typingDiv.remove();
        addMessage('assistant', INTERVIEW_QUESTIONS[idx]);
        // Speak the question aloud via TTS
        speakText(INTERVIEW_QUESTIONS[idx]);
        document.getElementById('chat-input').focus();
    }, 1200);
}

// ═══════════════════════════════════════════════════════════════════
//  TEXT-TO-SPEECH (Mira speaks)
// ═══════════════════════════════════════════════════════════════════
function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    // Strip markdown formatting
    const cleanText = text.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\n/g, '. ');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.95;
    utterance.pitch = 1.1;
    utterance.volume = 1;

    // Try to pick a female English voice
    const voices = window.speechSynthesis.getVoices();
    const femaleVoice = voices.find(v =>
        v.lang.startsWith('en') && (v.name.includes('Female') || v.name.includes('Zira') ||
            v.name.includes('Samantha') || v.name.includes('Google UK English Female') ||
            v.name.includes('Microsoft Zira'))
    ) || voices.find(v => v.lang.startsWith('en'));
    if (femaleVoice) utterance.voice = femaleVoice;

    // Show speaking indicator
    utterance.onstart = () => {
        isSpeaking = true;
        const indicator = document.getElementById('mira-speaking');
        if (indicator) indicator.style.display = 'flex';
    };
    utterance.onend = () => {
        isSpeaking = false;
        const indicator = document.getElementById('mira-speaking');
        if (indicator) indicator.style.display = 'none';
    };
    utterance.onerror = () => {
        isSpeaking = false;
        const indicator = document.getElementById('mira-speaking');
        if (indicator) indicator.style.display = 'none';
    };

    window.speechSynthesis.speak(utterance);
}

// Load voices (some browsers load them asynchronously)
if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}

// ═══════════════════════════════════════════════════════════════════
//  SPEECH-TO-TEXT (User speaks)
// ═══════════════════════════════════════════════════════════════════
function toggleMic() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert('Speech recognition is not supported in your browser. Please use Chrome or Edge.');
        return;
    }

    // Stop Mira from speaking if she is
    if (isSpeaking) window.speechSynthesis.cancel();

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    const input = document.getElementById('chat-input');
    const micBtn = document.getElementById('mic-btn');
    let finalTranscript = '';

    recognition.onresult = (event) => {
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + ' ';
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        input.value = finalTranscript + interimTranscript;
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopRecording();
    };

    recognition.onend = () => {
        // If still recording (not manually stopped), auto-send
        if (isRecording) {
            stopRecording();
            if (input.value.trim()) {
                sendMessage();
            }
        }
    };

    recognition.start();
    isRecording = true;
    micBtn.classList.add('recording');
    micBtn.innerHTML = '⏹';

    // Start audio visualizer
    startAudioVisualizer();

    // Start REC timer
    recSeconds = 0;
    const timerEl = document.getElementById('rec-timer');
    recTimerInterval = setInterval(() => {
        recSeconds++;
        const mins = Math.floor(recSeconds / 60).toString().padStart(2, '0');
        const secs = (recSeconds % 60).toString().padStart(2, '0');
        timerEl.textContent = `${mins}:${secs}`;
    }, 1000);
}

function stopRecording() {
    if (recognition) {
        isRecording = false;
        recognition.stop();
        recognition = null;
    }

    const micBtn = document.getElementById('mic-btn');
    micBtn.classList.remove('recording');
    micBtn.innerHTML = '🎤';

    // Stop visualizer
    stopAudioVisualizer();

    // Stop timer
    if (recTimerInterval) { clearInterval(recTimerInterval); recTimerInterval = null; }
}

// ═══════════════════════════════════════════════════════════════════
//  AUDIO WAVEFORM VISUALIZER
// ═══════════════════════════════════════════════════════════════════
async function startAudioVisualizer() {
    const visualizer = document.getElementById('audio-visualizer');
    const canvas = document.getElementById('audio-canvas');
    visualizer.style.display = 'flex';

    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(micStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const ctx = canvas.getContext('2d');

        function drawWaveform() {
            waveformAnimId = requestAnimationFrame(drawWaveform);
            analyser.getByteFrequencyData(dataArray);

            canvas.width = canvas.offsetWidth;
            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);

            const barWidth = (width / bufferLength) * 2;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * height;
                const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
                gradient.addColorStop(0, '#4F8EF7');
                gradient.addColorStop(1, '#9B6FFF');
                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
                x += barWidth;
            }
        }
        drawWaveform();
    } catch (err) {
        console.error('Audio visualizer error:', err);
    }
}

function stopAudioVisualizer() {
    if (waveformAnimId) { cancelAnimationFrame(waveformAnimId); waveformAnimId = null; }
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    const visualizer = document.getElementById('audio-visualizer');
    if (visualizer) visualizer.style.display = 'none';
}

function addMessage(role, text) {
    const messages = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;
    addMessage('user', text);
    interviewResponses.push(text);
    input.value = '';
    interviewIndex++;
    renderInterviewProgress();
    setTimeout(() => askQuestion(interviewIndex), 500);
}

function finishInterview() {
    const messages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;

    setTimeout(() => {
        typingDiv.remove();
        addMessage('assistant',
            "Thank you so much for sharing all of that with me. I really appreciate your openness. " +
            "Let me analyze your responses now. Please wait a moment...");
        document.getElementById('chat-input').disabled = true;
        document.getElementById('send-btn').disabled = true;
        stopFaceDetection();
        setTimeout(() => submitForAnalysis(), 2000);
    }, 1500);
}


// ═══════════════════════════════════════════════════════════════════
//  WEBCAM + FACE DETECTION
// ═══════════════════════════════════════════════════════════════════
async function loadFaceModels() {
    if (faceModelsLoaded) return;
    try {
        const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        faceModelsLoaded = true;
        console.log('✅ Face detection models loaded');
    } catch (err) {
        console.error('Face model load error:', err);
    }
}

async function toggleWebcam() {
    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('webcam-placeholder');
    const btn = document.getElementById('webcam-toggle');
    const overlay = document.getElementById('webcam-overlay');
    const liveExpr = document.getElementById('expression-live');

    if (webcamStream) {
        stopFaceDetection();
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
        video.style.display = 'none';
        overlay.style.display = 'none';
        liveExpr.style.display = 'none';
        placeholder.style.display = 'flex';
        placeholder.innerHTML = '<span class="icon">📷</span><span>Camera not active</span>';
        btn.textContent = 'Enable Camera';
    } else {
        try {
            placeholder.innerHTML = '<span class="icon">⏳</span><span>Starting camera...</span>';
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240, facingMode: 'user' }
            });
            video.srcObject = webcamStream;
            video.style.display = 'block';
            placeholder.style.display = 'none';
            btn.textContent = 'Disable Camera';

            if (typeof faceapi !== 'undefined') {
                liveExpr.style.display = 'block';
                document.getElementById('live-expression').textContent = 'Loading models...';
                await loadFaceModels();
                if (faceModelsLoaded) startFaceDetection();
            }
        } catch (err) {
            console.error('Webcam error:', err);
            placeholder.innerHTML = '<span class="icon">⚠️</span><span>Camera access denied</span>';
        }
    }
}

function startFaceDetection() {
    if (faceDetectionInterval) return;
    faceDetectionInterval = setInterval(async () => {
        const video = document.getElementById('webcam-video');
        if (!video || video.paused || video.ended || !faceModelsLoaded) return;
        totalDetectionAttempts++;
        try {
            const detection = await faceapi
                .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.4 }))
                .withFaceExpressions();

            if (detection) {
                faceDetectedCount++;
                const expr = detection.expressions;
                expressionHistory.push({
                    neutral: expr.neutral, happy: expr.happy, sad: expr.sad,
                    angry: expr.angry, fearful: expr.fearful, disgusted: expr.disgusted,
                    surprised: expr.surprised,
                });
                const sorted = Object.entries(expr).sort((a, b) => b[1] - a[1]);
                const dominant = sorted[0];
                const emojiMap = {
                    neutral: '😐', happy: '😊', sad: '😢', angry: '😠',
                    fearful: '😨', disgusted: '🤢', surprised: '😲'
                };
                document.getElementById('live-expression').textContent =
                    `${emojiMap[dominant[0]] || ''} ${dominant[0]} (${(dominant[1] * 100).toFixed(0)}%)`;
                drawFaceOverlay(detection);
            }
        } catch (err) { /* skip */ }
    }, 2500);
}

function drawFaceOverlay(detection) {
    const canvas = document.getElementById('webcam-overlay');
    const video = document.getElementById('webcam-video');
    if (!canvas || !video) return;
    canvas.style.display = 'block';
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const box = detection.detection.box;
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2;
    ctx.strokeRect(canvas.width - box.x - box.width, box.y, box.width, box.height);
}

function stopFaceDetection() {
    if (faceDetectionInterval) { clearInterval(faceDetectionInterval); faceDetectionInterval = null; }
}

function computeVisualAnalysis() {
    if (expressionHistory.length === 0) return null;
    const avg = { neutral: 0, happy: 0, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0 };
    for (const snap of expressionHistory) {
        for (const key of Object.keys(avg)) avg[key] += snap[key];
    }
    const n = expressionHistory.length;
    for (const key of Object.keys(avg)) avg[key] /= n;

    const flatAffect = avg.neutral;
    const sadScore = avg.sad + avg.fearful * 0.5;
    const happyScore = avg.happy;
    let visualProb = (flatAffect * 0.4 + sadScore * 0.4 + (1 - happyScore) * 0.2);
    visualProb = Math.max(0, Math.min(1, visualProb));

    return {
        averages: avg, flatAffect, visualProb,
        samplesCollected: n,
        faceDetectionRate: totalDetectionAttempts > 0 ? faceDetectedCount / totalDetectionAttempts : 0,
    };
}


// ═══════════════════════════════════════════════════════════════════
//  SUBMISSION & RESULTS
// ═══════════════════════════════════════════════════════════════════
async function submitForAnalysis() {
    showLoading('Analyzing your responses with AI models...');
    const interviewText = interviewResponses.join(' ');
    const visualData = computeVisualAnalysis();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phqAnswers, interviewText, visualData })
        });
        const data = await response.json();
        hideLoading();
        renderResults(data, visualData);
        goTo('results');
    } catch (err) {
        hideLoading();
        console.error('Prediction error:', err);
        addMessage('assistant', '❌ Sorry, there was an error analyzing your responses. Please try again.');
        document.getElementById('chat-input').disabled = false;
        document.getElementById('send-btn').disabled = false;
    }
}

function renderResults(data, visualData) {
    // ── Risk Gauge ─────────────────────────────────────────
    const prob = data.combined.probability;
    const pct = Math.round(prob * 100);
    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - prob);

    const gaugeFill = document.getElementById('gauge-fill');
    let strokeColor;
    if (prob >= 0.6) strokeColor = '#EF4444';
    else if (prob >= 0.4) strokeColor = '#F59E0B';
    else strokeColor = '#10B981';

    gaugeFill.style.stroke = strokeColor;
    setTimeout(() => { gaugeFill.style.strokeDashoffset = offset; }, 300);

    document.getElementById('gauge-value').textContent = pct + '%';
    const riskLabel = document.getElementById('risk-label');
    riskLabel.textContent = data.combined.riskLevel + ' Risk';
    riskLabel.style.color = strokeColor;

    // ── PHQ-8 Card ─────────────────────────────────────────
    const phqScore = data.phq.score;
    document.getElementById('phq-score-val').textContent = `${phqScore} / 24`;
    document.getElementById('phq-score-bar').style.width = (phqScore / 24 * 100) + '%';

    const phqBadge = document.getElementById('phq-badge');
    phqBadge.textContent = data.phq.severity;
    phqBadge.className = 'result-badge ' + (phqScore >= 10 ? 'high' : phqScore >= 5 ? 'moderate' : 'low');

    const sevDesc = {
        'Minimal': 'Your PHQ-8 score suggests minimal depressive symptoms.',
        'Mild': 'Your PHQ-8 score suggests mild symptoms. Consider monitoring.',
        'Moderate': 'Your score suggests moderate depression. Professional consultation recommended.',
        'Moderately Severe': 'Your score suggests moderately severe depression. Please seek help.',
        'Severe': 'Your score suggests severe depression. Contact a healthcare provider.'
    };
    document.getElementById('phq-severity-text').textContent = sevDesc[data.phq.severity] || '';

    // ── Text Analysis Card ─────────────────────────────────
    const textProb = data.text.probability;
    const textPct = Math.round(textProb * 100);
    document.getElementById('text-prob-val').textContent = textPct + '%';
    document.getElementById('text-prob-bar').style.width = textPct + '%';

    const textBadge = document.getElementById('text-badge');
    if (textProb >= 0.6) { textBadge.textContent = 'Elevated'; textBadge.className = 'result-badge high'; }
    else if (textProb >= 0.4) { textBadge.textContent = 'Moderate'; textBadge.className = 'result-badge moderate'; }
    else { textBadge.textContent = 'Normal'; textBadge.className = 'result-badge low'; }

    fetchSentiment();

    // ── Visual Analysis Card ───────────────────────────────
    renderVisualResults(visualData);
}

function renderVisualResults(visualData) {
    const badge = document.getElementById('visual-badge');
    const flatBar = document.getElementById('visual-flat-bar');
    const flatVal = document.getElementById('visual-flat-val');
    const chart = document.getElementById('expression-chart');
    const note = document.getElementById('visual-note');

    if (!visualData || visualData.samplesCollected === 0) {
        badge.textContent = 'No Data';
        badge.className = 'result-badge moderate';
        flatVal.textContent = 'N/A';
        chart.innerHTML = '<p style="font-size:0.85rem; color:var(--text-4)">Camera was not enabled during the interview. Enable the camera next time for facial expression analysis.</p>';
        note.textContent = '';
        return;
    }

    const flatPct = Math.round(visualData.flatAffect * 100);
    flatVal.textContent = flatPct + '%';
    flatBar.style.width = flatPct + '%';

    const vp = visualData.visualProb;
    if (vp >= 0.6) { badge.textContent = 'Elevated'; badge.className = 'result-badge high'; }
    else if (vp >= 0.4) { badge.textContent = 'Moderate'; badge.className = 'result-badge moderate'; }
    else { badge.textContent = 'Normal'; badge.className = 'result-badge low'; }

    const emojiMap = {
        neutral: '😐', happy: '😊', sad: '😢', angry: '😠',
        fearful: '😨', disgusted: '🤢', surprised: '😲'
    };
    const colorMap = {
        neutral: '#7B8BA0', happy: '#10B981', sad: '#4F8EF7',
        angry: '#EF4444', fearful: '#F59E0B', disgusted: '#9B6FFF', surprised: '#06d6a0'
    };

    const avg = visualData.averages;
    chart.innerHTML = Object.entries(avg)
        .sort((a, b) => b[1] - a[1])
        .map(([name, value]) => `
      <div class="expression-row">
        <span class="expression-emoji">${emojiMap[name]}</span>
        <span class="expression-name">${name.charAt(0).toUpperCase() + name.slice(1)}</span>
        <div class="expression-track">
          <div class="expression-bar" style="width:${value * 100}%; background:${colorMap[name]}"></div>
        </div>
        <span class="expression-pct">${(value * 100).toFixed(0)}%</span>
      </div>`
        ).join('');

    note.textContent = `Based on ${visualData.samplesCollected} facial snapshots captured during the interview (${Math.round(visualData.faceDetectionRate * 100)}% face detection rate).`;
}

async function fetchSentiment() {
    const text = interviewResponses.join(' ');
    try {
        const res = await fetch('/api/analyze-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        if (data.sentiment) renderSentiment(data.sentiment);
    } catch (e) { console.error('Sentiment error:', e); }
}

function renderSentiment(sentiment) {
    const container = document.getElementById('sentiment-bars');
    const items = [
        { label: 'Negative', value: sentiment.neg, color: '#EF4444' },
        { label: 'Neutral', value: sentiment.neu, color: '#7B8BA0' },
        { label: 'Positive', value: sentiment.pos, color: '#10B981' },
    ];
    container.innerHTML = items.map(item => `
    <div class="sentiment-row">
      <span class="sentiment-label">${item.label}</span>
      <div class="sentiment-track">
        <div class="sentiment-fill" style="width:${item.value * 100}%; background:${item.color}"></div>
      </div>
      <span class="sentiment-value">${(item.value * 100).toFixed(0)}%</span>
    </div>`).join('');
}


// ═══════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════
function showLoading(text) {
    document.getElementById('loading-text').textContent = text || 'Loading...';
    document.getElementById('loading').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function restart() {
    phqAnswers = Array(8).fill(-1);
    phqIndex = 0;
    interviewIndex = 0;
    interviewResponses = [];
    expressionHistory = [];
    faceDetectedCount = 0;
    totalDetectionAttempts = 0;
    document.getElementById('chat-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    stopFaceDetection();
    stopRecording();
    stopAudioVisualizer();
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    goTo('landing');
    // Restore main nav
    document.getElementById('nav-links-main').style.display = 'flex';
    document.getElementById('nav-steps').style.display = 'none';
}

// Keyboard shortcut: 0-3 for PHQ answers, Enter for next
document.addEventListener('keydown', (e) => {
    if (currentSection === 'phq' && ['0', '1', '2', '3'].includes(e.key)) {
        selectPhqOption(parseInt(e.key));
    }
});
