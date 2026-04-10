// ================================================================
//  SENTIRA — Frontend Application Logic
//  Flow: Landing → Setup → Interview → PHQ-8 → Results
//  Features: TTS, STT, Audio Recording + Playback, Smooth Face Detection
// ================================================================

// ── State ──────────────────────────────────────────────────────
let currentSection = 'landing';
let phqAnswers = Array(8).fill(-1);
let phqIndex = 0;
let interviewIndex = 0;
let interviewResponses = [];
let webcamStream = null;

// Face detection — smooth & dynamic
let faceModelsLoaded = false;
let faceDetectionInterval = null;
let expressionHistory = [];
let faceDetectedCount = 0;
let totalDetectionAttempts = 0;
let smoothedExpressions = null; // for smooth interpolation
let lastFaceBox = null; // for smooth box animation
let faceOverlayAnimId = null;

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

// Audio feature analysis state
let audioFeatureHistory = [];
let audioSilenceMs = 0;
let audioSpeechMs = 0;
let audioSpeechSegments = 0;
let audioLastWasSilent = true;
let audioLastSampleTime = 0;
let audioCollectIntervalId = null;

// Audio recording + playback
let mediaRecorder = null;
let recordedChunks = [];
let lastRecordingBlob = null;
let lastRecordingUrl = null;

// Shared audio stream manager — avoids multiple permission requests
let _sharedAudioStream = null;
async function getSharedAudioStream() {
    if (_sharedAudioStream && _sharedAudioStream.active) return _sharedAudioStream;
    _sharedAudioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return _sharedAudioStream;
}
function releaseSharedAudioStream() {
    if (_sharedAudioStream) { _sharedAudioStream.getTracks().forEach(t => t.stop()); _sharedAudioStream = null; }
}

// ── Toast Notification System ─────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { info: 'ℹ️', success: '✅', warning: '⚠️', error: '❌' };
    toast.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span><span class="toast-msg">${message}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
        toast.classList.remove('show');
        toast.addEventListener('transitionend', () => toast.remove());
    }, duration);
}

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
    "Hey there! I'm **Mira**, your SENTIRA assistant. I'm just going to ask you a few casual questions — no right or wrong answers, just be yourself.\n\n**So, how are you doing today? How's life been?**",
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
    speakText("Hi! I'm Mira, your SENTIRA assistant. I'll be guiding you through a short conversation. Can you hear me clearly?", () => {
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
        showToast('Speech recognition is not supported in your browser. Please use Chrome or Edge.', 'warning', 5000);
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

// ── Media Recorder (for playback) — uses shared audio stream ─────
async function startMediaRecorder() {
    try {
        const stream = await getSharedAudioStream();
        recordedChunks = [];

        // Pick the first supported mimeType (cross-browser: Chrome=webm, Firefox=ogg, Safari=mp4)
        const mimeType = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            'audio/mp4',
            ''
        ].find(t => t === '' || MediaRecorder.isTypeSupported(t));

        const options = mimeType ? { mimeType } : {};
        mediaRecorder = new MediaRecorder(stream, options);
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
            if (recordedChunks.length > 0) {
                const blobType = mimeType || 'audio/webm';
                lastRecordingBlob = new Blob(recordedChunks, { type: blobType });
                if (lastRecordingUrl) URL.revokeObjectURL(lastRecordingUrl);
                lastRecordingUrl = URL.createObjectURL(lastRecordingBlob);
                const playBtn = document.getElementById('playback-btn');
                if (playBtn) playBtn.style.display = 'flex';
            }
        };
        mediaRecorder.start(500); // Request data every 500ms so chunks are captured
    } catch (err) {
        console.warn('MediaRecorder unavailable:', err.message);
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
        micStream = await getSharedAudioStream();
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(micStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const ctx = canvas.getContext('2d');

        audioLastSampleTime = Date.now();
        if (audioCollectIntervalId) clearInterval(audioCollectIntervalId);
        audioCollectIntervalId = setInterval(() => {
            collectAudioFeatures(dataArray, bufferLength);
        }, 500);

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
        showToast('Microphone access failed. Voice analysis unavailable.', 'warning');
    }
}

// ═══════════════════════════════════════════════════════════════════
//  AUDIO FEATURE EXTRACTION — Real-time voice cue analysis
// ═══════════════════════════════════════════════════════════════════
function collectAudioFeatures(dataArray, bufferLength) {
    if (!analyser) return;
    analyser.getByteFrequencyData(dataArray);

    const now = Date.now();
    const dt = now - audioLastSampleTime;
    audioLastSampleTime = now;

    // RMS Energy
    let sumSq = 0;
    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 255;
        sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / bufferLength);

    // Spectral centroid (weighted mean of frequency bins)
    let weightedSum = 0, totalWeight = 0;
    for (let i = 0; i < bufferLength; i++) {
        weightedSum += i * dataArray[i];
        totalWeight += dataArray[i];
    }
    const centroid = totalWeight > 0 ? weightedSum / totalWeight : 0;

    // Dominant frequency bin
    let maxBin = 0, maxVal = 0;
    for (let i = 0; i < bufferLength; i++) {
        if (dataArray[i] > maxVal) { maxVal = dataArray[i]; maxBin = i; }
    }

    // Silence vs speech detection (RMS threshold)
    const SILENCE_THRESH = 0.06;
    const isSilent = rms < SILENCE_THRESH;
    if (isSilent) {
        audioSilenceMs += dt;
        if (!audioLastWasSilent) { /* transition to silence */ }
    } else {
        audioSpeechMs += dt;
        if (audioLastWasSilent) {
            audioSpeechSegments++; // new speech burst
        }
    }
    audioLastWasSilent = isSilent;

    audioFeatureHistory.push({
        energy: rms,
        centroid: centroid,
        dominantBin: maxBin,
        isSilent: isSilent,
        timestamp: now
    });
}

function computeAudioAnalysis() {
    if (audioFeatureHistory.length < 3) return null;

    const n = audioFeatureHistory.length;
    let sumEnergy = 0, sumCentroid = 0;
    const energyValues = [];

    for (const f of audioFeatureHistory) {
        sumEnergy += f.energy;
        sumCentroid += f.centroid;
        energyValues.push(f.energy);
    }

    const avgEnergy = sumEnergy / n;
    const avgCentroid = sumCentroid / n;

    // Energy variability (std dev)
    let sumSqDiff = 0;
    for (const e of energyValues) sumSqDiff += (e - avgEnergy) ** 2;
    const energyStd = Math.sqrt(sumSqDiff / n);

    const totalTimeMs = audioSpeechMs + audioSilenceMs;
    const pauseRatio = totalTimeMs > 0 ? audioSilenceMs / totalTimeMs : 0;
    const speechRate = totalTimeMs > 0 ? (audioSpeechSegments / (totalTimeMs / 1000)) : 0;
    const avgPauseDur = audioSpeechSegments > 0 ? (audioSilenceMs / 1000) / Math.max(1, audioSpeechSegments) : 0;

    // Depression scoring: low energy + low centroid + high pause ratio + low variability (monotone)
    // Each sub-score maps to 0-1 range
    const energyScore = Math.max(0, Math.min(1, 1 - (avgEnergy / 0.25)));      // lower energy → higher score
    const centroidScore = Math.max(0, Math.min(1, 1 - (avgCentroid / 64)));     // lower centroid → higher score
    const pauseScore = Math.max(0, Math.min(1, pauseRatio * 1.5));              // more pauses → higher score
    const monotoneScore = Math.max(0, Math.min(1, 1 - (energyStd / 0.12)));    // less variation → higher score

    let audioProb = (
        energyScore * 0.30 +
        centroidScore * 0.25 +
        pauseScore * 0.25 +
        monotoneScore * 0.20
    );
    audioProb = Math.max(0, Math.min(1, audioProb));

    return {
        avgEnergy: avgEnergy,
        avgCentroid: avgCentroid,
        energyVariability: energyStd,
        pauseRatio: pauseRatio,
        speechRate: speechRate,
        avgPauseDuration: avgPauseDur,
        audioProb: audioProb,
        samplesCollected: n,
        totalTimeSec: totalTimeMs / 1000
    };
}

function stopAudioVisualizer() {
    if (waveformAnimId) { cancelAnimationFrame(waveformAnimId); waveformAnimId = null; }
    if (audioCollectIntervalId) { clearInterval(audioCollectIntervalId); audioCollectIntervalId = null; }
    micStream = null; // don't stop shared stream here
    if (audioContext) { audioContext.close(); audioContext = null; analyser = null; }
    const vis = document.getElementById('audio-visualizer');
    if (vis) vis.style.display = 'none';
}

// ── Background audio analysis — runs throughout entire interview ───
// Collects audio features silently even when the user types instead of speaking.
let _bgAudioCtx = null;
let _bgAnalyser = null;
let _bgCollectId = null;

async function startBackgroundAudioAnalysis() {
    if (_bgCollectId) return; // already running
    try {
        const stream = await getSharedAudioStream();
        _bgAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = _bgAudioCtx.createMediaStreamSource(stream);
        _bgAnalyser = _bgAudioCtx.createAnalyser();
        _bgAnalyser.fftSize = 256;
        source.connect(_bgAnalyser);

        const bufLen = _bgAnalyser.frequencyBinCount;
        const dataArr = new Uint8Array(bufLen);
        audioLastSampleTime = Date.now();

        _bgCollectId = setInterval(() => {
            if (!_bgAnalyser) return;
            // If the visualizer is also running (mic btn pressed), skip to avoid double-counting
            if (audioCollectIntervalId) return;
            _bgAnalyser.getByteFrequencyData(dataArr);
            collectAudioFeatures(dataArr, bufLen);
        }, 500);

        console.log('[Audio] Background analysis started');
    } catch (err) {
        console.warn('[Audio] Background analysis unavailable:', err.message);
    }
}

function stopBackgroundAudioAnalysis() {
    if (_bgCollectId) { clearInterval(_bgCollectId); _bgCollectId = null; }
    if (_bgAudioCtx) { _bgAudioCtx.close(); _bgAudioCtx = null; _bgAnalyser = null; }
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

    // Reset audio analysis state for fresh session
    audioFeatureHistory = [];
    audioSilenceMs = 0;
    audioSpeechMs = 0;
    audioSpeechSegments = 0;
    audioLastWasSilent = true;
    audioLastSampleTime = 0;
    if (audioCollectIntervalId) { clearInterval(audioCollectIntervalId); audioCollectIntervalId = null; }

    // Transfer camera from setup to interview
    transferCameraToInterview();
    renderInterviewProgress();

    // Auto-start background audio analysis so data is collected even if user types
    startBackgroundAudioAnalysis();

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
    const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
    const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 15000));
    try {
        await Promise.race([
            Promise.all([
                faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
                faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
            ]),
            timeout
        ]);
        faceModelsLoaded = true;
        showToast('Face detection models loaded', 'success', 2000);
    } catch (err) {
        showToast('Face models failed to load. Facial analysis unavailable.', 'warning', 5000);
        console.error('Face model load error:', err);
    }
}

function startFaceDetection() {
    if (faceDetectionInterval) return;
    // Fast detection interval (800ms) for smooth, dynamic updates
    faceDetectionInterval = setInterval(async () => {
        const video = document.getElementById('webcam-video');
        if (!video || video.paused || video.ended || !faceModelsLoaded) return;
        totalDetectionAttempts++;
        try {
            const detection = await faceapi
                .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.35 }))
                .withFaceExpressions();
            const liveEl = document.getElementById('live-expression');
            if (detection) {
                faceDetectedCount++;
                const expr = detection.expressions;
                expressionHistory.push({
                    neutral: expr.neutral, happy: expr.happy, sad: expr.sad,
                    angry: expr.angry, fearful: expr.fearful, disgusted: expr.disgusted,
                    surprised: expr.surprised,
                });
                // Smooth interpolation — blend new values with previous
                const alpha = 0.4; // smoothing factor (0=no update, 1=instant)
                if (!smoothedExpressions) {
                    smoothedExpressions = { ...expr };
                } else {
                    for (const k of Object.keys(smoothedExpressions)) {
                        smoothedExpressions[k] = smoothedExpressions[k] * (1 - alpha) + (expr[k] || 0) * alpha;
                    }
                }
                const sorted = Object.entries(smoothedExpressions).sort((a, b) => b[1] - a[1]);
                const dominant = sorted[0];
                const emojiMap = { neutral: '😐', happy: '😊', sad: '😢', angry: '😠', fearful: '😨', disgusted: '🤢', surprised: '😲' };
                if (liveEl) liveEl.textContent = `${emojiMap[dominant[0]] || ''} ${dominant[0]} (${(dominant[1] * 100).toFixed(0)}%)`;
                // Animate face overlay smoothly
                animateFaceOverlay(detection);
            } else {
                if (liveEl && faceDetectedCount === 0) liveEl.textContent = 'No face detected';
            }
        } catch (err) { /* skip frame */ }
    }, 800);
}

// ── Smooth animated face overlay ─────────────────────────────────
let _targetBox = null;
let _currentBox = null;
function animateFaceOverlay(detection) {
    const canvas = document.getElementById('webcam-overlay');
    const video = document.getElementById('webcam-video');
    if (!canvas || !video) return;
    canvas.style.display = 'block';
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const box = detection.detection.box;
    _targetBox = { x: canvas.width - box.x - box.width, y: box.y, w: box.width, h: box.height };
    if (!_currentBox) _currentBox = { ..._targetBox };
    if (!faceOverlayAnimId) drawSmoothOverlay(canvas);
}

function drawSmoothOverlay(canvas) {
    const ctx = canvas.getContext('2d');
    function frame() {
        if (!_targetBox || !_currentBox) { faceOverlayAnimId = null; return; }
        faceOverlayAnimId = requestAnimationFrame(frame);
        // Lerp towards target
        const s = 0.15;
        _currentBox.x += (_targetBox.x - _currentBox.x) * s;
        _currentBox.y += (_targetBox.y - _currentBox.y) * s;
        _currentBox.w += (_targetBox.w - _currentBox.w) * s;
        _currentBox.h += (_targetBox.h - _currentBox.h) * s;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Glowing rounded rect
        const r = 8;
        const { x, y, w, h } = _currentBox;
        ctx.beginPath();
        ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#10B981';
        ctx.shadowBlur = 12;
        ctx.stroke();
        ctx.shadowBlur = 0;
        // Corner accents
        const cl = 12;
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#34D399';
        [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]].forEach(([cx,cy,dx,dy]) => {
            ctx.beginPath(); ctx.moveTo(cx, cy + dy*cl); ctx.lineTo(cx, cy); ctx.lineTo(cx + dx*cl, cy); ctx.stroke();
        });
        // Dominant expression label near box
        if (smoothedExpressions) {
            const sorted = Object.entries(smoothedExpressions).sort((a, b) => b[1] - a[1]);
            const emojiMap = { neutral: '😐', happy: '😊', sad: '😢', angry: '😠', fearful: '😨', disgusted: '🤢', surprised: '😲' };
            ctx.font = '13px "DM Sans", sans-serif';
            ctx.fillStyle = '#10B981';
            ctx.fillText(`${emojiMap[sorted[0][0]]||''} ${sorted[0][0]}`, x, y - 8);
        }
    }
    frame();
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

    // Stop all background analysis
    stopFaceDetection();
    stopBackgroundAudioAnalysis();
    stopAudioVisualizer();
    hideMiniWebcam();

    const interviewText = interviewResponses.join(' ');
    const visualData = computeVisualAnalysis();
    const audioData = computeAudioAnalysis();

    console.log(`[Submit] audio samples=${audioData ? audioData.samplesCollected : 0}, visual samples=${visualData ? visualData.samplesCollected : 0}`);

    // Stop webcam stream
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phqAnswers, interviewText, visualData, audioData })
        });
        const data = await response.json();
        hideLoading();
        renderResults(data, visualData, audioData);
        goTo('results');
    } catch (err) {
        hideLoading();
        console.error('Prediction error:', err);
        showToast('Error analyzing responses. Please try again.', 'error');
    }
}

function renderResults(data, visualData, audioData) {
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
    renderAudioResults(audioData);
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

function renderAudioResults(audioData) {
    const badge = document.getElementById('audio-badge');
    const probBar = document.getElementById('audio-prob-bar');
    const probVal = document.getElementById('audio-prob-val');
    const energyBar = document.getElementById('audio-energy-bar');
    const energyVal = document.getElementById('audio-energy-val');
    const pauseBar = document.getElementById('audio-pause-bar');
    const pauseVal = document.getElementById('audio-pause-val');
    const rateVal = document.getElementById('audio-rate-val');
    const note = document.getElementById('audio-note');

    if (!audioData || audioData.samplesCollected < 3) {
        if (badge) { badge.textContent = 'No Data'; badge.className = 'result-badge moderate'; }
        if (probVal) probVal.textContent = 'N/A';
        if (energyVal) energyVal.textContent = 'N/A';
        if (pauseVal) pauseVal.textContent = 'N/A';
        if (rateVal) rateVal.textContent = 'N/A';
        if (note) note.textContent = 'Microphone was not used. Use the mic button next time for voice analysis.';
        return;
    }

    const ap = audioData.audioProb;
    const apPct = Math.round(ap * 100);
    if (probVal) probVal.textContent = apPct + '%';
    if (probBar) probBar.style.width = apPct + '%';

    if (ap >= 0.6) { badge.textContent = 'Elevated'; badge.className = 'result-badge high'; }
    else if (ap >= 0.4) { badge.textContent = 'Moderate'; badge.className = 'result-badge moderate'; }
    else { badge.textContent = 'Normal'; badge.className = 'result-badge low'; }

    // Energy (scale 0-0.3 to 0-100%)
    const energyPct = Math.min(100, Math.round((audioData.avgEnergy / 0.3) * 100));
    if (energyVal) energyVal.textContent = energyPct + '%';
    if (energyBar) energyBar.style.width = energyPct + '%';

    // Pause ratio
    const pausePct = Math.round(audioData.pauseRatio * 100);
    if (pauseVal) pauseVal.textContent = pausePct + '%';
    if (pauseBar) pauseBar.style.width = pausePct + '%';

    // Speech rate
    if (rateVal) rateVal.textContent = audioData.speechRate.toFixed(1) + ' segments/s';

    if (note) {
        note.textContent = `Based on ${audioData.samplesCollected} audio samples over ${audioData.totalTimeSec.toFixed(0)}s of recording.`;
    }
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
    // Reset audio analysis state
    audioFeatureHistory = [];
    audioSilenceMs = 0;
    audioSpeechMs = 0;
    audioSpeechSegments = 0;
    audioLastWasSilent = true;
    audioLastSampleTime = 0;
    if (audioCollectIntervalId) { clearInterval(audioCollectIntervalId); audioCollectIntervalId = null; }
    if (lastRecordingUrl) { URL.revokeObjectURL(lastRecordingUrl); lastRecordingUrl = null; }
    document.getElementById('chat-input').disabled = false;
    document.getElementById('send-btn').disabled = false;
    stopFaceDetection();
    if (faceOverlayAnimId) { cancelAnimationFrame(faceOverlayAnimId); faceOverlayAnimId = null; }
    _targetBox = null; _currentBox = null; smoothedExpressions = null;
    hideMiniWebcam();
    stopRecording();
    stopAudioVisualizer();
    stopBackgroundAudioAnalysis();
    releaseSharedAudioStream();
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

// ── Export Results as Image ───────────────────────────────────────
function exportResults() {
    const resultsEl = document.getElementById('results');
    if (!resultsEl) return;
    showToast('Preparing your report...', 'info', 2000);
    // Use canvas-based screenshot via html2canvas if available, else simple text export
    if (typeof html2canvas !== 'undefined') {
        html2canvas(resultsEl, { backgroundColor: '#0a0a0f', scale: 2 }).then(canvas => {
            const link = document.createElement('a');
            link.download = 'SENTIRA_Report.png';
            link.href = canvas.toDataURL();
            link.click();
            showToast('Report downloaded!', 'success');
        });
    } else {
        // Fallback: export as text summary
        const gaugeVal = document.getElementById('gauge-value')?.textContent || '';
        const riskLabel = document.getElementById('risk-label')?.textContent || '';
        const phqScore = document.getElementById('phq-score-val')?.textContent || '';
        const textProb = document.getElementById('text-prob-val')?.textContent || '';
        const summary = `SENTIRA Depression Screening Report\n${'='.repeat(40)}\nRisk Score: ${gaugeVal}\nRisk Level: ${riskLabel}\nPHQ-8 Score: ${phqScore}\nText Analysis: ${textProb}\n\nDate: ${new Date().toLocaleDateString()}\n\nDisclaimer: This is a screening tool, not a clinical diagnosis.`;
        const blob = new Blob([summary], { type: 'text/plain' });
        const link = document.createElement('a');
        link.download = 'SENTIRA_Report.txt';
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
        showToast('Report downloaded!', 'success');
    }
}
