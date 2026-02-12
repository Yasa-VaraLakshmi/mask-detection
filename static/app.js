const startWebcamBtn = document.getElementById('startWebcam');
const stopWebcamBtn = document.getElementById('stopWebcam');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const voiceButton = document.getElementById('voiceButton');
const askMaskBtn = document.getElementById('askMaskBtn');
const voiceStatus = document.getElementById('voiceStatus');
const voiceHint = document.getElementById('voiceHint');
const chat = document.getElementById('chat');

const resultLabel = document.getElementById('resultLabel');
const resultConfidence = document.getElementById('resultConfidence');
const faceCount = document.getElementById('faceCount');

let stream = null;
let webcamTimer = null;
let inFlight = false;
let listening = false;

function addBubble(text, role) {
  const div = document.createElement('div');
  div.className = `bubble ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function speakBrowser(text) {
  if (!window.speechSynthesis) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

function updateDetectionStatus(results) {
  const count = Array.isArray(results) ? results.length : 0;
  faceCount.textContent = String(count);

  if (!count) {
    document.body.dataset.detection = 'none';
    resultLabel.textContent = 'No Face';
    resultLabel.className = 'status-value neutral';
    resultConfidence.textContent = '-';
    return;
  }

  const top = results[0];
  const rawLabel = String(top.label || '').toLowerCase();
  const confidence = Number(top.confidence || 0);

  if (rawLabel === 'with_mask') {
    document.body.dataset.detection = 'mask';
    resultLabel.textContent = 'Mask';
    resultLabel.className = 'status-value good';
  } else {
    document.body.dataset.detection = 'no-mask';
    resultLabel.textContent = 'No Mask';
    resultLabel.className = 'status-value bad';
  }

  resultConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
}

function setVoiceState(stateText) {
  voiceStatus.textContent = stateText;
}

async function startWebcam() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    startWebcamBtn.disabled = true;
    stopWebcamBtn.disabled = false;
    webcamTimer = setInterval(captureFrame, 500);
    addBubble('Webcam started.', 'assistant');
  } catch (error) {
    addBubble(`Webcam error: ${error.message}`, 'assistant');
  }
}

function stopWebcam() {
  if (webcamTimer) {
    clearInterval(webcamTimer);
    webcamTimer = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  startWebcamBtn.disabled = false;
  stopWebcamBtn.disabled = true;
  updateDetectionStatus([]);
  addBubble('Webcam stopped.', 'assistant');
}

async function captureFrame() {
  if (!stream || inFlight || !video.videoWidth || !video.videoHeight) return;
  inFlight = true;

  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

  try {
    const res = await fetch('/predict_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });

    if (!res.ok) {
      throw new Error(`Server error ${res.status}`);
    }

    const data = await res.json();
    updateDetectionStatus(data.results || []);
  } catch (error) {
    setVoiceState('Server error');
    addBubble(`Prediction error: ${error.message}`, 'assistant');
  } finally {
    inFlight = false;
  }
}

function captureCurrentFrameDataUrl() {
  if (!stream || !video.videoWidth || !video.videoHeight) return null;
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function sendCommand(text, includeFrame = true) {
  addBubble(text, 'user');
  setVoiceState('Sending...');

  try {
    const body = { text };
    if (includeFrame) {
      const frameData = captureCurrentFrameDataUrl();
      if (frameData) {
        body.image = frameData;
      }
    }

    const res = await fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    const data = await res.json();
    const reply = data.reply || data.error || 'No response from assistant.';
    addBubble(reply, 'assistant');
    speakBrowser(reply);
    setVoiceState('Idle');
  } catch (error) {
    addBubble(`Assistant error: ${error.message}`, 'assistant');
    setVoiceState('Idle');
  }
}

startWebcamBtn.addEventListener('click', startWebcam);
stopWebcamBtn.addEventListener('click', stopWebcam);
askMaskBtn.addEventListener('click', () => sendCommand('is the person wearing a mask', true));

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    listening = true;
    setVoiceState('Listening...');
    voiceButton.textContent = 'Listening';
    voiceButton.disabled = true;
  };

  recognition.onresult = async (event) => {
    const text = event.results?.[0]?.[0]?.transcript?.trim();
    if (!text) {
      addBubble('No speech detected.', 'assistant');
      setVoiceState('Idle');
      return;
    }
    await sendCommand(text.toLowerCase(), true);
  };

  recognition.onerror = (event) => {
    addBubble(`Voice error: ${event.error}`, 'assistant');
    setVoiceState('Idle');
  };

  recognition.onend = () => {
    listening = false;
    voiceButton.textContent = 'Start Listening';
    voiceButton.disabled = false;
    if (voiceStatus.textContent === 'Listening...') {
      setVoiceState('Idle');
    }
  };
} else {
  voiceButton.disabled = true;
  voiceHint.textContent = 'Speech recognition is not supported in this browser. Use Ask Mask Status.';
}

voiceButton.addEventListener('click', () => {
  if (!recognition || listening) return;
  recognition.start();
});
