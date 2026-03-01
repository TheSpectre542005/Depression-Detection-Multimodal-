// ================================================================
//  MindScan — Frontend Application Logic
//  Flow: Landing → Setup → Interview → PHQ-8 → Results
//  Features: TTS, STT, Audio Recording + Playback, Face Detection
// ================================================================

// ── State ──────────────────────────────────────────────────────
let currentSection = 'landing';
let phqAnswers = Array(8).fill(-1);
let phqIndex = 0;
let interviewIndex = 0;
let interviewResponses = [];
let webcamStream = null;

// Face detection
let faceModelsLoaded = false;
let faceDetectionInterval = null;
let expressionHistory = [];
let faceDetectedCount = 0;
let totalDetectionAttempts = 0;

// Audio TTS + STT
let isSpeaking = false;
let isRecording = false;
let recognition = null;
let audioContext = null;
let analyser = null;
let micStream = null;
let waveformAnimId = null;
let recTimerInterval = null;
let recSeconds = 0;

// Audio recording + playback
let mediaRecorder = null;
let recordedChunks = [];
let lastRecordingBlob = null;
let lastRecordingUrl = null;

// Setup screen state
let setupCamStream = null;
let setupMicStream = null;
let setupMicCtx = null;
let setupMicAnalyser = null;
let setupMicAnimId = null;

// ────────────────────────────────────────────────────────────────
//  INIT
// ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initWaveforms();
    initNavScroll();
    initCounters();
    // Preload voices
    if ('speechSynthesis' in window) window.speechSynthesis.getVoices();
});

// ── Scroll-triggered animations ─────────────────────────────────
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                if (entry.target.querySelector('.stagger')) {
                    entry.target.querySelectorAll('.stagger > *').forEach((child, i) => {
                        child.style.setProperty('--i', i);
                        setTimeout(() => child.classList.add('visible'), i * 100);
                    });
                }
            }
        });
    }, { threshold: 0.15 });
    document.querySelectorAll('.animate-up').forEach(el => observer.observe(el));
    document.querySelectorAll('.stagger').forEach(el => {
        const obs = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    Array.from(entry.target.children).forEach((child, i) =>
                        setTimeout(() => child.classList.add('visible'), i * 120));
                }
            });
        }, { threshold: 0.1 });
        obs.observe(el);
    });
}

function initWaveforms() {
    const colors = { 'wave-text': '#4F8EF7', 'wave-audio': '#9B6FFF', 'wave-visual': '#10B981' };
    Object.keys(colors).forEach(id => {
        const container = document.getElementById(id);
        if (!container) return;
        for (let i = 0; i < 40; i++) {
            const bar = document.createElement('span');
            bar.style.height = (15 + Math.random() * 85) + '%';
            bar.style.background = colors[id];
            bar.style.animationDelay = (Math.random() * 1.5) + 's';
            bar.style.animationDuration = (1 + Math.random()) + 's';
            container.appendChild(bar);
        }
    });
}

function initNavScroll() {
    const nav = document.getElementById('main-nav');
    window.addEventListener('scroll', () => {
        nav.classList.toggle('scrolled', window.scrollY > 50);
    });
}

function initCounters() {
    const counters = document.querySelectorAll('.stat-number');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.dataset.animated) {
                entry.target.dataset.animated = 'true';
                const target = parseFloat(entry.target.dataset.target);
                const suffix = entry.target.dataset.suffix || '';
                const isFloat = target % 1 !== 0;
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
    const step = ts => {
        if (!start) start = ts;
        const p = Math.min((ts - start) / duration, 1);
        el.textContent = (isFloat ? (easeOut(p) * target).toFixed(2) : Math.floor(easeOut(p) * target)) + suffix;
        if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}


// ═══════════════════════════════════════════════════════════════════
//  PHQ-8 DATA
// ═══════════════════════════════════════════════════════════════════
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


// ═══════════════════════════════════════════════════════════════════
//  INTERVIEW QUESTIONS — Simple, conversational, depression-focused
// ═══════════════════════════════════════════════════════════════════
const INTERVIEW_QUESTIONS = [
    "Hey there! I'm **Mira**, your MindScan assistant. I'm just going to ask you a few casual questions — no right or wrong answers, just be yourself.\n\n**So, how are you doing today? How's life been?**",
    "Thanks for sharing. **What do you usually do for fun?** Have you been enjoying those things lately, or not so much?",
    "Got it. **How have you been sleeping?** Like, do you sleep well or has it been tough?",
    "**How's your energy been?** Do you feel tired a lot, or are you generally okay?",
    "**Can you focus on things easily?** Like work, studying, watching something — or do you zone out a lot?",
    "**How are things with your friends and family?** Do you feel close to people, or more alone lately?",
    "**When you think about the future**, how does it feel? Exciting, stressful, or kinda blank?",
    "Last one — **is there anything that's been really bothering you lately?** Anything weighing on your mind?"
];


// ═══════════════════════════════════════════════════════════════════
//  NAVIGATION — New flow: Landing → Setup → Interview → PHQ → Results
// ═══════════════════════════════════════════════════════════════════
function goTo(sectionId) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');

    const navLinks = document.getElementById('nav-links-main');
    const navSteps = document.getElementById('nav-steps');

    if (sectionId === 'landing') {
        navLinks.style.display = 'flex';
        navSteps.style.display = 'none';
    } else {
        navLinks.style.display = 'none';
        navSteps.style.display = 'flex';
    }

    // Step dots — 5 steps now
    const steps = ['landing', 'setup', 'interview', 'phq', 'results'];
    steps.forEach((s, i) => {
        const dot = document.getElementById('dot-' + s);
        if (!dot) return;
        dot.classList.remove('active', 'done');
        const targetIdx = steps.indexOf(sectionId);
        if (i < targetIdx) dot.classList.add('done');
        if (i === targetIdx) dot.classList.add('active');
    });

    currentSection = sectionId;
    window.scrollTo(0, 0);

    if (sectionId === 'setup') initSetupScreen();
    if (sectionId === 'interview') startInterview();
    if (sectionId === 'phq') {
        renderPhqQuestion();
        showMiniWebcam();   // Keep facial analysis visible during PHQ
    }
    if (sectionId === 'results') {
        hideMiniWebcam();
    }
}


// ═══════════════════════════════════════════════════════════════════
//  SETUP SCREEN — Camera + Mic + Voice Check
// ═══════════════════════════════════════════════════════════════════
function initSetupScreen() {
    // Reset statuses
    document.getElementById('cam-status').textContent = 'Not checked';
    document.getElementById('cam-status').className = 'setup-status';
    document.getElementById('mic-status').textContent = 'Not checked';
    document.getElementById('mic-status').className = 'setup-status';
    document.getElementById('voice-status').textContent = 'Not tested';
    document.getElementById('voice-status').className = 'setup-status';
}

async function setupToggleCamera() {
    const video = document.getElementById('setup-cam-video');
    const placeholder = document.getElementById('setup-cam-placeholder');
    const btn = document.getElementById('setup-cam-btn');
    const status = document.getElementById('cam-status');

    if (setupCamStream) {
        setupCamStream.getTracks().forEach(t => t.stop());
        setupCamStream = null;
        video.style.display = 'none';
        placeholder.style.display = 'flex';
        btn.textContent = 'Enable Camera';
        status.textContent = 'Not checked';
        status.className = 'setup-status';
    } else {
        try {
            placeholder.innerHTML = '<span class="icon">⏳</span><span>Starting camera...</span>';
            setupCamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240, facingMode: 'user' }
            });
            video.srcObject = setupCamStream;
            video.style.display = 'block';
            placeholder.style.display = 'none';
            btn.textContent = 'Disable Camera';
            status.textContent = '✓ Working';
            status.className = 'setup-status ok';
        } catch (err) {
            placeholder.innerHTML = '<span class="icon">⚠️</span><span>Camera access denied</span>';
            status.textContent = '✗ Failed';
            status.className = 'setup-status fail';
        }
    }
}

async function setupToggleMic() {
    const canvas = document.getElementById('setup-mic-canvas');
    const text = document.getElementById('setup-mic-text');
    const btn = document.getElementById('setup-mic-btn');
    const status = document.getElementById('mic-status');

    if (setupMicStream) {
        if (setupMicAnimId) cancelAnimationFrame(setupMicAnimId);
        setupMicStream.getTracks().forEach(t => t.stop());
        setupMicStream = null;
        if (setupMicCtx) { setupMicCtx.close(); setupMicCtx = null; }
        btn.textContent = 'Test Microphone';
        text.textContent = 'Click below to test your mic';
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    } else {
        try {
            text.textContent = 'Listening... speak now to see the waveform';
            setupMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setupMicCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = setupMicCtx.createMediaStreamSource(setupMicStream);
            setupMicAnalyser = setupMicCtx.createAnalyser();
            setupMicAnalyser.fftSize = 256;
            source.connect(setupMicAnalyser);
            btn.textContent = 'Stop Test';
            status.textContent = '✓ Working';
            status.className = 'setup-status ok';

            const bufferLength = setupMicAnalyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            const ctx = canvas.getContext('2d');

            function drawSetupMic() {
                setupMicAnimId = requestAnimationFrame(drawSetupMic);
                setupMicAnalyser.getByteFrequencyData(dataArray);
                canvas.width = canvas.offsetWidth;
                const w = canvas.width, h = canvas.height;
                ctx.clearRect(0, 0, w, h);
                const barW = (w / bufferLength) * 2.5;
                let x = 0;
                for (let i = 0; i < bufferLength; i++) {
                    const barH = (dataArray[i] / 255) * h;
                    const grad = ctx.createLinearGradient(0, h, 0, h - barH);
                    grad.addColorStop(0, '#4F8EF7');
                    grad.addColorStop(1, '#9B6FFF');
                    ctx.fillStyle = grad;
                    ctx.fillRect(x, h - barH, barW - 1, barH);
                    x += barW;
                }
            }
            drawSetupMic();
        } catch (err) {
            text.textContent = 'Microphone access denied';
            status.textContent = '✗ Failed';
            status.className = 'setup-status fail';
        }
    }
}

function testMiraVoice() {
    const status = document.getElementById('voice-status');
    speakText("Hi! I'm Mira, your MindScan assistant. I'll be guiding you through a short conversation. Can you hear me clearly?", () => {
        status.textContent = '✓ Played';
        status.className = 'setup-status ok';
    });
}


// ═══════════════════════════════════════════════════════════════════
//  TEXT-TO-SPEECH — Better voice selection
// ═══════════════════════════════════════════════════════════════════
function speakText(text, onEndCallback) {
    if (!('speechSynthesis' in window)) { if (onEndCallback) onEndCallback(); return; }
    window.speechSynthesis.cancel();

    const cleanText = text.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\n/g, '. ');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.9;
    utterance.pitch = 1.05;
    utterance.volume = 1;

    // Voice priority list — most natural sounding
    const voices = window.speechSynthesis.getVoices();
    const voicePriority = [
        'Microsoft Jenny', 'Google UK English Female', 'Samantha',
        'Microsoft Zira', 'Karen', 'Moira', 'Tessa', 'Fiona',
        'Google US English', 'Victoria', 'Alex'
    ];

    let selectedVoice = null;
    for (const name of voicePriority) {
        selectedVoice = voices.find(v => v.name.includes(name));
        if (selectedVoice) break;
    }
    if (!selectedVoice) selectedVoice = voices.find(v => v.lang.startsWith('en'));
    if (selectedVoice) utterance.voice = selectedVoice;

    utterance.onstart = () => {
        isSpeaking = true;
        const ind = document.getElementById('mira-speaking');
        if (ind) ind.style.display = 'flex';
    };
    const onDone = () => {
        isSpeaking = false;
        const ind = document.getElementById('mira-speaking');
        if (ind) ind.style.display = 'none';
        if (onEndCallback) onEndCallback();
    };
    utterance.onend = onDone;
    utterance.onerror = onDone;

    window.speechSynthesis.speak(utterance);
}

if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}


// ═══════════════════════════════════════════════════════════════════
//  SPEECH-TO-TEXT + AUDIO RECORDING + PLAYBACK
// ═══════════════════════════════════════════════════════════════════
function toggleMic() {
    if (isRecording) stopRecording();
    else startRecording();
}

function startRecording() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert('Speech recognition is not supported in your browser. Please use Chrome or Edge.');
        return;
    }

    if (isSpeaking) window.speechSynthesis.cancel();

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 3;

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
        if (event.error !== 'no-speech') stopRecording();
    };

    recognition.onend = () => {
        // Don't auto-send — let user review and press send
        if (isRecording) {
            isRecording = false;
            const micBtn = document.getElementById('mic-btn');
            micBtn.classList.remove('recording');
            micBtn.innerHTML = '🎤';
            stopAudioVisualizer();
            stopMediaRecorder();
            if (recTimerInterval) { clearInterval(recTimerInterval); recTimerInterval = null; }
        }
    };

    recognition.start();
    isRecording = true;
    micBtn.classList.add('recording');
    micBtn.innerHTML = '⏹';

    // Start audio visualizer + media recorder for playback
    startAudioVisualizer();
    startMediaRecorder();

    // Timer
    recSeconds = 0;
    const timerEl = document.getElementById('rec-timer');
    recTimerInterval = setInterval(() => {
        recSeconds++;
        timerEl.textContent = `${String(Math.floor(recSeconds / 60)).padStart(2, '0')}:${String(recSeconds % 60).padStart(2, '0')}`;
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
    stopAudioVisualizer();
    stopMediaRecorder();
    if (recTimerInterval) { clearInterval(recTimerInterval); recTimerInterval = null; }
}

// ── Media Recorder (for playback) ────────────────────────────────
async function startMediaRecorder() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
            if (recordedChunks.length > 0) {
                lastRecordingBlob = new Blob(recordedChunks, { type: 'audio/webm' });
                if (lastRecordingUrl) URL.revokeObjectURL(lastRecordingUrl);
                lastRecordingUrl = URL.createObjectURL(lastRecordingBlob);
                // Show playback button
                const playBtn = document.getElementById('playback-btn');
                if (playBtn) playBtn.style.display = 'flex';
            }
            stream.getTracks().forEach(t => t.stop());
        };
        mediaRecorder.start();
    } catch (err) {
        console.error('MediaRecorder error:', err);
    }
}

function stopMediaRecorder() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}

function playbackRecording() {
    if (!lastRecordingUrl) return;
    const audio = new Audio(lastRecordingUrl);
    const btn = document.getElementById('playback-btn');
    btn.innerHTML = '⏹';
    btn.classList.add('playing');
    audio.play();
    audio.onended = () => {
        btn.innerHTML = '🔊';
        btn.classList.remove('playing');
    };
}


// ═══════════════════════════════════════════════════════════════════
//  AUDIO WAVEFORM VISUALIZER
// ═══════════════════════════════════════════════════════════════════
async function startAudioVisualizer() {
    const visualizer = document.getElementById('audio-visualizer');
    const canvas = document.getElementById('audio-canvas');
    if (!visualizer || !canvas) return;
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

        function draw() {
            waveformAnimId = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);
            canvas.width = canvas.offsetWidth;
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            const barW = (w / bufferLength) * 2;
            let x = 0;
            for (let i = 0; i < bufferLength; i++) {
                const barH = (dataArray[i] / 255) * h;
                const grad = ctx.createLinearGradient(0, h, 0, h - barH);
                grad.addColorStop(0, '#4F8EF7');
                grad.addColorStop(1, '#9B6FFF');
                ctx.fillStyle = grad;
                ctx.fillRect(x, h - barH, barW - 1, barH);
                x += barW;
            }
        }
        draw();
    } catch (err) {
        console.error('Audio visualizer error:', err);
    }
}

function stopAudioVisualizer() {
    if (waveformAnimId) { cancelAnimationFrame(waveformAnimId); waveformAnimId = null; }
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    const vis = document.getElementById('audio-visualizer');
    if (vis) vis.style.display = 'none';
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

    document.getElementById('phq-progress').style.width = ((phqIndex + 1) / 8 * 100) + '%';
    document.getElementById('phq-progress-label').textContent = `${phqIndex + 1} / 8`;
    document.getElementById('phq-back-btn').disabled = phqIndex === 0;
    updatePhqNextBtn();
}

function selectPhqOption(value) { phqAnswers[phqIndex] = value; renderPhqQuestion(); }

function updatePhqNextBtn() {
    const btn = document.getElementById('phq-next-btn');
    btn.disabled = phqAnswers[phqIndex] < 0;
    btn.textContent = phqIndex === 7 ? 'Get My Results →' : 'Next →';
}

function phqNext() {
    if (phqAnswers[phqIndex] < 0) return;
    if (phqIndex < 7) { phqIndex++; renderPhqQuestion(); }
    else submitForAnalysis();  // Last PHQ question → submit everything
}

function phqBack() { if (phqIndex > 0) { phqIndex--; renderPhqQuestion(); } }


// ═══════════════════════════════════════════════════════════════════
//  VIRTUAL INTERVIEW
// ═══════════════════════════════════════════════════════════════════
function startInterview() {
    interviewIndex = 0;
    interviewResponses = [];
    document.getElementById('chat-messages').innerHTML = '';
    document.getElementById('chat-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    const playBtn = document.getElementById('playback-btn');
    if (playBtn) playBtn.style.display = 'none';

    // Transfer camera from setup to interview
    transferCameraToInterview();
    renderInterviewProgress();
    setTimeout(() => askQuestion(0), 600);
}

function transferCameraToInterview() {
    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('webcam-placeholder');

    if (setupCamStream) {
        // Use the camera stream from setup
        webcamStream = setupCamStream;
        setupCamStream = null;
        video.srcObject = webcamStream;
        video.style.display = 'block';
        placeholder.style.display = 'none';

        // Load face models and start detection
        initFaceAnalysis();
    } else {
        // Auto-request camera if user didn't enable it in setup
        autoStartCamera();
    }
}

async function autoStartCamera() {
    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('webcam-placeholder');
    try {
        placeholder.style.display = 'flex';
        placeholder.innerHTML = '<span class="icon">⏳</span><span>Starting camera...</span>';
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 320, height: 240, facingMode: 'user' }
        });
        video.srcObject = webcamStream;
        video.style.display = 'block';
        placeholder.style.display = 'none';
        initFaceAnalysis();
    } catch (err) {
        console.warn('Camera auto-start failed:', err);
        placeholder.innerHTML = '<span class="icon">📷</span><span>Camera not available</span>';
    }
}

function initFaceAnalysis() {
    if (typeof faceapi !== 'undefined') {
        const liveExpr = document.getElementById('expression-live');
        liveExpr.style.display = 'block';
        document.getElementById('live-expression').textContent = 'Loading models...';
        loadFaceModels().then(() => {
            if (faceModelsLoaded) startFaceDetection();
        });
    }
}

function renderInterviewProgress() {
    const list = document.getElementById('q-progress-list');
    const labels = ['General wellbeing', 'Interests & hobbies', 'Sleep quality', 'Energy levels',
        'Focus & concentration', 'Social connection', 'Future outlook', 'Worries & concerns'];
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
        speakText(INTERVIEW_QUESTIONS[idx]);
        document.getElementById('chat-input').focus();
    }, 1200);
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
    // Hide playback button after sending
    const playBtn = document.getElementById('playback-btn');
    if (playBtn) playBtn.style.display = 'none';
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
            "Thank you so much for chatting with me! I really appreciate you being so open. " +
            "Now we just have a quick questionnaire — 8 short questions. Almost done!");
        speakText("Thank you so much for chatting with me! Now let's do a quick questionnaire.");
        document.getElementById('chat-input').disabled = true;
        document.getElementById('send-btn').disabled = true;
        // NOTE: Do NOT stop face detection here — let it sustain through PHQ-8
        // Go to PHQ after a pause
        setTimeout(() => goTo('phq'), 4000);
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
                const emojiMap = { neutral: '😐', happy: '😊', sad: '😢', angry: '😠', fearful: '😨', disgusted: '🤢', surprised: '😲' };
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

// ── Mini-webcam PiP during PHQ-8 ─────────────────────────────────
function showMiniWebcam() {
    // Only show if we have an active webcam stream
    if (!webcamStream) return;
    let mini = document.getElementById('mini-webcam');
    if (!mini) {
        mini = document.createElement('div');
        mini.id = 'mini-webcam';
        mini.className = 'mini-webcam-pip';
        mini.innerHTML = `
            <video id="mini-webcam-video" autoplay playsinline muted></video>
            <div class="mini-webcam-label" id="mini-webcam-label">
                <span class="mini-rec-dot"></span>
                <span id="mini-expression-text">Analyzing...</span>
            </div>
        `;
        document.body.appendChild(mini);
    }
    const miniVideo = document.getElementById('mini-webcam-video');
    miniVideo.srcObject = webcamStream;
    mini.style.display = 'block';
    // Update mini expression text from the main detection loop
    if (!mini._updateInterval) {
        mini._updateInterval = setInterval(() => {
            const mainExpr = document.getElementById('live-expression');
            const miniExpr = document.getElementById('mini-expression-text');
            if (mainExpr && miniExpr) {
                miniExpr.textContent = mainExpr.textContent;
            }
        }, 3000);
    }
}

function hideMiniWebcam() {
    const mini = document.getElementById('mini-webcam');
    if (mini) {
        if (mini._updateInterval) {
            clearInterval(mini._updateInterval);
            mini._updateInterval = null;
        }
        mini.style.display = 'none';
    }
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
    return { averages: avg, flatAffect, visualProb, samplesCollected: n, faceDetectionRate: totalDetectionAttempts > 0 ? faceDetectedCount / totalDetectionAttempts : 0 };
}


// ═══════════════════════════════════════════════════════════════════
//  SUBMISSION & RESULTS
// ═══════════════════════════════════════════════════════════════════
async function submitForAnalysis() {
    showLoading('Analyzing your responses with AI models...');

    // Stop face detection NOW — we've collected enough data
    stopFaceDetection();
    hideMiniWebcam();

    const interviewText = interviewResponses.join(' ');
    const visualData = computeVisualAnalysis();

    // Stop webcam stream
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }

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
        alert('Error analyzing responses. Please try again.');
    }
}

function renderResults(data, visualData) {
    const prob = data.combined.probability;
    const pct = Math.round(prob * 100);
    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - prob);
    const gaugeFill = document.getElementById('gauge-fill');
    let strokeColor = prob >= 0.6 ? '#EF4444' : prob >= 0.4 ? '#F59E0B' : '#10B981';
    gaugeFill.style.stroke = strokeColor;
    setTimeout(() => { gaugeFill.style.strokeDashoffset = offset; }, 300);
    document.getElementById('gauge-value').textContent = pct + '%';
    const riskLabel = document.getElementById('risk-label');
    riskLabel.textContent = data.combined.riskLevel + ' Risk';
    riskLabel.style.color = strokeColor;

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

    const textProb = data.text.probability;
    const textPct = Math.round(textProb * 100);
    document.getElementById('text-prob-val').textContent = textPct + '%';
    document.getElementById('text-prob-bar').style.width = textPct + '%';
    const textBadge = document.getElementById('text-badge');
    if (textProb >= 0.6) { textBadge.textContent = 'Elevated'; textBadge.className = 'result-badge high'; }
    else if (textProb >= 0.4) { textBadge.textContent = 'Moderate'; textBadge.className = 'result-badge moderate'; }
    else { textBadge.textContent = 'Normal'; textBadge.className = 'result-badge low'; }
    fetchSentiment();
    renderVisualResults(visualData);
}

function renderVisualResults(visualData) {
    const badge = document.getElementById('visual-badge');
    const flatBar = document.getElementById('visual-flat-bar');
    const flatVal = document.getElementById('visual-flat-val');
    const chart = document.getElementById('expression-chart');
    const note = document.getElementById('visual-note');
    if (!visualData || visualData.samplesCollected === 0) {
        badge.textContent = 'No Data'; badge.className = 'result-badge moderate';
        flatVal.textContent = 'N/A';
        chart.innerHTML = '<p style="font-size:0.85rem; color:var(--text-4)">Camera was not enabled. Enable it in setup next time.</p>';
        note.textContent = ''; return;
    }
    const flatPct = Math.round(visualData.flatAffect * 100);
    flatVal.textContent = flatPct + '%';
    flatBar.style.width = flatPct + '%';
    const vp = visualData.visualProb;
    if (vp >= 0.6) { badge.textContent = 'Elevated'; badge.className = 'result-badge high'; }
    else if (vp >= 0.4) { badge.textContent = 'Moderate'; badge.className = 'result-badge moderate'; }
    else { badge.textContent = 'Normal'; badge.className = 'result-badge low'; }

    const emojiMap = { neutral: '😐', happy: '😊', sad: '😢', angry: '😠', fearful: '😨', disgusted: '🤢', surprised: '😲' };
    const colorMap = { neutral: '#7B8BA0', happy: '#10B981', sad: '#4F8EF7', angry: '#EF4444', fearful: '#F59E0B', disgusted: '#9B6FFF', surprised: '#06d6a0' };
    chart.innerHTML = Object.entries(visualData.averages).sort((a, b) => b[1] - a[1]).map(([name, value]) =>
        `<div class="expression-row"><span class="expression-emoji">${emojiMap[name]}</span><span class="expression-name">${name.charAt(0).toUpperCase() + name.slice(1)}</span><div class="expression-track"><div class="expression-bar" style="width:${value * 100}%; background:${colorMap[name]}"></div></div><span class="expression-pct">${(value * 100).toFixed(0)}%</span></div>`
    ).join('');
    note.textContent = `Based on ${visualData.samplesCollected} snapshots (${Math.round(visualData.faceDetectionRate * 100)}% detection rate).`;
}

async function fetchSentiment() {
    try {
        const res = await fetch('/api/analyze-text', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: interviewResponses.join(' ') })
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
    container.innerHTML = items.map(item =>
        `<div class="sentiment-row"><span class="sentiment-label">${item.label}</span><div class="sentiment-track"><div class="sentiment-fill" style="width:${item.value * 100}%; background:${item.color}"></div></div><span class="sentiment-value">${(item.value * 100).toFixed(0)}%</span></div>`
    ).join('');
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
    lastRecordingBlob = null;
    if (lastRecordingUrl) { URL.revokeObjectURL(lastRecordingUrl); lastRecordingUrl = null; }
    document.getElementById('chat-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    stopFaceDetection();
    hideMiniWebcam();
    stopRecording();
    stopAudioVisualizer();
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
    if (setupCamStream) { setupCamStream.getTracks().forEach(t => t.stop()); setupCamStream = null; }
    goTo('landing');
    document.getElementById('nav-links-main').style.display = 'flex';
    document.getElementById('nav-steps').style.display = 'none';
}

document.addEventListener('keydown', (e) => {
    if (currentSection === 'phq' && ['0', '1', '2', '3'].includes(e.key)) {
        selectPhqOption(parseInt(e.key));
    }
});
