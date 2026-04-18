import type {ChangeEvent} from 'react';
import {useEffect, useRef, useState} from 'react';
import {Activity, ArrowRight, Github, RotateCcw} from 'lucide-react';

const VIDEO_URL =
  'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260328_083109_283f3553-e28f-428b-a723-d639c617eb2b.mp4';
const GITHUB_URL = 'https://github.com/Tuancoolboy/gpt2-disease';
const HUGGING_FACE_URL = 'https://huggingface.co/sanim05/GPT2-disease_text_generation';
const MODEL_ID = 'sanim05/GPT2-disease_text_generation';
const STORAGE_KEY = 'aethera-disease-settings';
const GENERATION_TIMEOUT_MS = 120_000;

type ExamplePreset = {
  topic: string;
  aliases: string[];
  prompt: string;
};

const examplePresets = {
  'Tim mạch': {
    topic: 'bệnh tim',
    aliases: ['benh tim', 'tim mach', 'benh mach vanh', 'tim'],
    prompt: 'bệnh tim là',
  },
  'Hô hấp': {
    topic: 'viêm phổi',
    aliases: ['viem phoi', 'ho hap', 'hen phe quan', 'hen suyen'],
    prompt: 'viêm phổi là',
  },
  'Tiêu hóa': {
    topic: 'đau dạ dày',
    aliases: ['dau da day', 'tieu hoa', 'viem loet da day', 'da day'],
    prompt: 'đau dạ dày là',
  },
  'Thần kinh': {
    topic: 'tai biến mạch máu não',
    aliases: ['tai bien mach mau nao', 'than kinh', 'dot quy', 'dot quy nao'],
    prompt: 'tai biến mạch máu não là',
  },
  'Nội tiết': {
    topic: 'tiểu đường tuýp 2',
    aliases: ['tieu duong', 'noi tiet', 'duong huyet cao', 'tieu duong tuyp 2'],
    prompt: 'tiểu đường tuýp 2 là',
  },
  'Da liễu': {
    topic: 'viêm da cơ địa',
    aliases: ['viem da co dia', 'da lieu', 'ngua da', 'cham'],
    prompt: 'viêm da cơ địa là',
  },
  'Nhi khoa': {
    topic: 'sốt xuất huyết ở trẻ em',
    aliases: ['sot xuat huyet', 'nhi khoa', 'tre em', 'sot xuat huyet o tre em'],
    prompt: 'sốt xuất huyết ở trẻ em là',
  },
  'Bệnh mạn tính': {
    topic: 'tăng huyết áp',
    aliases: ['tang huyet ap', 'benh man tinh', 'cao huyet ap', 'huyet ap cao'],
    prompt: 'tăng huyết áp là',
  },
} satisfies Record<string, ExamplePreset>;

type ExampleKey = keyof typeof examplePresets;
type GenerationSettings = {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
};
type SettingKey = keyof GenerationSettings;
type BackendStatus = 'checking' | 'online' | 'offline';
type GenerationResponse = {
  generated_text?: string;
  error?: string;
};

const exampleKeys = Object.keys(examplePresets) as ExampleKey[];
const defaultExampleKey: ExampleKey = 'Hô hấp';

const DEFAULT_SETTINGS: GenerationSettings = {
  maxNewTokens: 120,
  temperature: 0.8,
  topP: 0.95,
  repetitionPenalty: 1.1,
};

const generationControls: Array<{
  key: SettingKey;
  label: string;
  min: number;
  max: number;
  step: number;
}> = [
  {key: 'maxNewTokens', label: 'Độ dài', min: 48, max: 320, step: 8},
  {key: 'temperature', label: 'Giọng văn', min: 0.2, max: 1.3, step: 0.05},
  {key: 'topP', label: 'Độ bám ý', min: 0.6, max: 1, step: 0.01},
  {key: 'repetitionPenalty', label: 'Giảm lặp', min: 1, max: 1.4, step: 0.05},
];

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function normalizeSettings(candidate?: Partial<GenerationSettings>): GenerationSettings {
  return {
    maxNewTokens: Math.round(
      clamp(candidate?.maxNewTokens ?? DEFAULT_SETTINGS.maxNewTokens, 48, 320),
    ),
    temperature: clamp(candidate?.temperature ?? DEFAULT_SETTINGS.temperature, 0.2, 1.3),
    topP: clamp(candidate?.topP ?? DEFAULT_SETTINGS.topP, 0.6, 1),
    repetitionPenalty: clamp(
      candidate?.repetitionPenalty ?? DEFAULT_SETTINGS.repetitionPenalty,
      1,
      1.4,
    ),
  };
}

function normalizeKey(value: string) {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/g, 'd')
    .replace(/Đ/g, 'D')
    .toLowerCase()
    .trim();
}

function findPresetByTopic(topic: string) {
  const normalized = normalizeKey(topic);
  return Object.values(examplePresets).find((preset) =>
    preset.aliases.some((alias) => normalized.includes(alias)),
  );
}

function findPresetKeyByTopic(topic: string) {
  const normalized = normalizeKey(topic);
  return exampleKeys.find((key) =>
    examplePresets[key].aliases.some((alias) => normalized.includes(alias)),
  );
}

function buildOpeningText(topic: string) {
  const cleanTopic = topic.trim() || examplePresets[defaultExampleKey].topic;
  const matchedPreset = findPresetByTopic(cleanTopic);

  if (matchedPreset) {
    return matchedPreset.prompt;
  }

  return `${cleanTopic} là`;
}

function describeLength(value: number) {
  if (value <= 96) return 'Ngắn';
  if (value <= 176) return 'Vừa';
  return 'Dài';
}

function describeTemperature(value: number) {
  if (value <= 0.55) return 'Ổn định';
  if (value <= 0.9) return 'Tự nhiên';
  return 'Linh hoạt';
}

function describeTopP(value: number) {
  if (value <= 0.75) return 'Chặt';
  if (value <= 0.9) return 'Cân bằng';
  return 'Thoáng';
}

function describeRepetition(value: number) {
  if (value <= 1.05) return 'Nhẹ';
  if (value <= 1.2) return 'Vừa';
  return 'Mạnh';
}

function describeControlValue(key: SettingKey, value: number) {
  if (key === 'maxNewTokens') return describeLength(value);
  if (key === 'temperature') return describeTemperature(value);
  if (key === 'topP') return describeTopP(value);
  return describeRepetition(value);
}

async function parseGenerationResponse(response: Response): Promise<GenerationResponse> {
  const contentType = response.headers.get('content-type') ?? '';

  if (contentType.includes('application/json')) {
    return (await response.json()) as GenerationResponse;
  }

  const rawText = (await response.text()).trim();
  const text = rawText || 'Empty response from backend.';

  if (response.status === 404) {
    return {
      error: `Backend route not found (404). Received: ${text.slice(0, 160)}`,
    };
  }

  return {
    error: `Backend returned a non-JSON response. Received: ${text.slice(0, 160)}`,
  };
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoOpacity, setVideoOpacity] = useState(0);
  const [selectedExample, setSelectedExample] = useState<ExampleKey>(defaultExampleKey);
  const [topic, setTopic] = useState(examplePresets[defaultExampleKey].topic);
  const [prompt, setPrompt] = useState(examplePresets[defaultExampleKey].prompt);
  const [isPromptCustomized, setIsPromptCustomized] = useState(false);
  const [generatedText, setGeneratedText] = useState('');
  const [generationError, setGenerationError] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const [settings, setSettings] = useState<GenerationSettings>(() => {
    const raw = window.localStorage.getItem(STORAGE_KEY);

    if (!raw) {
      return DEFAULT_SETTINGS;
    }

    try {
      return normalizeSettings(JSON.parse(raw) as Partial<GenerationSettings>);
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
      return DEFAULT_SETTINGS;
    }
  });

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let rafId = 0;

    const updateOpacity = () => {
      if (video.duration) {
        const currentTime = video.currentTime;
        const duration = video.duration;
        const fadeTime = 0.6;

        let opacity = 1;

        if (currentTime < fadeTime) {
          opacity = currentTime / fadeTime;
        } else if (currentTime > duration - fadeTime) {
          opacity = (duration - currentTime) / fadeTime;
        }

        setVideoOpacity(Math.max(0, Math.min(1, opacity)));
      }

      rafId = requestAnimationFrame(updateOpacity);
    };

    rafId = requestAnimationFrame(updateOpacity);
    return () => cancelAnimationFrame(rafId);
  }, []);

  useEffect(() => {
    let active = true;

    const checkBackend = async () => {
      try {
        const response = await fetch('/api/health');

        if (!response.ok) {
          throw new Error('Backend chưa sẵn sàng');
        }

        if (active) {
          setBackendStatus('online');
        }
      } catch {
        if (active) {
          setBackendStatus('offline');
        }
      }
    };

    checkBackend();
    return () => {
      active = false;
    };
  }, []);

  const handleVideoEnded = () => {
    const video = videoRef.current;
    if (!video) return;

    setVideoOpacity(0);
    setTimeout(() => {
      video.currentTime = 0;
      video.play().catch(() => {});
    }, 100);
  };

  const handleSelectExample = (key: ExampleKey) => {
    const preset = examplePresets[key];
    setSelectedExample(key);
    setTopic(preset.topic);
    setPrompt(preset.prompt);
    setIsPromptCustomized(false);
    setGenerationError('');
  };

  const handleApplyTopic = () => {
    setPrompt(buildOpeningText(topic));
    const matchedKey = findPresetKeyByTopic(topic);

    if (matchedKey) {
      setSelectedExample(matchedKey);
    }

    setIsPromptCustomized(false);
    setGenerationError('');
  };

  const handleRestoreExample = () => {
    const preset = examplePresets[selectedExample];
    setTopic(preset.topic);
    setPrompt(preset.prompt);
    setIsPromptCustomized(false);
    setGenerationError('');
  };

  const handleTopicChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextTopic = event.target.value;
    const matchedKey = findPresetKeyByTopic(nextTopic);

    setTopic(nextTopic);

    if (matchedKey) {
      setSelectedExample(matchedKey);
    }

    if (!isPromptCustomized) {
      setPrompt(buildOpeningText(nextTopic));
    }
  };

  const handlePromptChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(event.target.value);
    setIsPromptCustomized(true);
  };

  const handleSettingChange =
    (key: SettingKey) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const rawValue = Number(event.target.value);

      if (Number.isNaN(rawValue)) {
        return;
      }

      const control = generationControls.find((item) => item.key === key);
      if (!control) {
        return;
      }

      const nextValue =
        key === 'maxNewTokens'
          ? Math.round(clamp(rawValue, control.min, control.max))
          : clamp(rawValue, control.min, control.max);

      setSettings((current) => ({
        ...current,
        [key]: nextValue,
      }));
    };

  const handleResetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
  };

  const handleGenerate = async () => {
    if (isGenerating) {
      return;
    }

    if (!prompt.trim()) {
      setGenerationError('Bạn cần nhập nội dung mở đầu trước khi chạy model.');
      return;
    }

    setIsGenerating(true);
    setGeneratedText('');
    setGenerationError('');
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), GENERATION_TIMEOUT_MS);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        signal: controller.signal,
        body: JSON.stringify({
          prompt,
          max_new_tokens: settings.maxNewTokens,
          temperature: settings.temperature,
          top_p: settings.topP,
          repetition_penalty: settings.repetitionPenalty,
        }),
      });

      const data = await parseGenerationResponse(response);

      if (!response.ok || !data.generated_text) {
        throw new Error(data.error ?? 'Không thể tạo nội dung từ model.');
      }

      setGeneratedText(data.generated_text);
      setBackendStatus('online');
    } catch (error) {
      setGeneratedText('');
      setBackendStatus('offline');
      if (error instanceof DOMException && error.name === 'AbortError') {
        setGenerationError(
          'Model đang phản hồi quá lâu. Nếu đây là lần chạy đầu, hãy chờ backend tải model xong rồi thử lại.',
        );
        return;
      }

      setGenerationError(
        error instanceof Error
          ? error.message
          : 'Không thể kết nối backend hoặc model chưa khởi động.',
      );
    } finally {
      window.clearTimeout(timeoutId);
      setIsGenerating(false);
    }
  };

  return (
    <div className="aethera-shell">
      <div className="aethera-background">
        <video
          ref={videoRef}
          src={VIDEO_URL}
          autoPlay
          muted
          playsInline
          onEnded={handleVideoEnded}
          className="aethera-video"
          style={{opacity: videoOpacity}}
          referrerPolicy="no-referrer"
        />
        <div className="aethera-video-overlay" />
        <div className="aethera-noise" />
      </div>

      <div className="aethera-content">
        <header className="aethera-nav">
          <div className="aethera-brand">
            <span className="aethera-brand-mark">
              <Activity size={18} />
            </span>
            <div>
              <p className="aethera-brand-title">Academic Disease Generation</p>
              <p className="aethera-brand-subtitle">GPT-2 Vietnamese Health from Scratch</p>
            </div>
          </div>

          <div className="aethera-nav-links">
            <span>{MODEL_ID}</span>
            <a href={HUGGING_FACE_URL} target="_blank" rel="noreferrer">
              Hugging Face
            </a>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">
              <Github size={15} />
              GitHub
            </a>
          </div>
        </header>

        <main className="aethera-hero">
          <p className="aethera-eyebrow">Vietnamese Disease Information Generation</p>

          <h1 className="aethera-title">
            Generate Vietnamese disease information.
          </h1>

          <p className="aethera-summary">
            Công cụ này dùng mô hình GPT-2 tiếng Việt để tiếp tục đoạn mở đầu về bệnh, giúp tạo
            nội dung giới thiệu, triệu chứng, nguyên nhân và biến chứng theo cách đọc dễ hiểu.
            Bạn có thể chọn ví dụ bệnh, chỉnh nội dung mở đầu và sinh kết quả.
          </p>

          <div className="aethera-examples">
            {exampleKeys.map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => handleSelectExample(key)}
                className={`aethera-pill ${selectedExample === key ? 'is-active' : ''}`}
              >
                Ví dụ {key}
              </button>
            ))}
          </div>

          <div className="aethera-flow">
            <section className="aethera-form">
              <div className="aethera-compose">
                <label className="aethera-label" htmlFor="topic">
                  Chủ đề bệnh
                </label>
                <input
                  id="topic"
                  value={topic}
                  onChange={handleTopicChange}
                  className="aethera-input"
                  placeholder="Ví dụ: bệnh tim, viêm phổi, đau dạ dày"
                />

                <label className="aethera-label aethera-label-gap" htmlFor="prompt">
                  Prompt
                </label>
                <textarea
                  id="prompt"
                  value={prompt}
                  onChange={handlePromptChange}
                  className="aethera-textarea"
                  placeholder="Đoạn văn mở đầu sẽ được gửi sang backend để model viết tiếp..."
                />

                <div className="aethera-actions">
                  <button type="button" onClick={handleApplyTopic} className="aethera-cta">
                    Tạo lại prompt
                    <ArrowRight size={16} />
                  </button>
                  <button
                    type="button"
                    onClick={handleGenerate}
                    className="aethera-ghost"
                    disabled={isGenerating}
                  >
                    {isGenerating ? 'Đang sinh nội dung...' : 'Sinh nội dung bệnh'}
                  </button>
                  <button type="button" onClick={handleRestoreExample} className="aethera-ghost">
                    Khôi phục ví dụ
                  </button>
                </div>
              </div>
            </section>

            <section className="aethera-controls">
              <div className="aethera-inline-meta">
                <span>Ví dụ hiện tại: {selectedExample}</span>
                <span>Độ dài: {describeLength(settings.maxNewTokens)}</span>
                <span>Model: GPT-2 Vietnamese Health</span>
              </div>

              <section className="aethera-output">
                <div className="aethera-output-head">
                  <span>Nội dung sinh ra</span>
                  <span>{MODEL_ID}</span>
                </div>

                {generationError ? (
                  <p className="aethera-error">{generationError}</p>
                ) : null}

                <div className="aethera-output-text">
                  {isGenerating ? (
                    <p className="aethera-loading-copy">
                      Model đang tạo nội dung từ prompt. Lần chạy đầu có thể mất 20 đến 60 giây,
                      nhất là khi backend đang chạy bằng CPU.
                    </p>
                  ) : generatedText ? (
                    <p className="whitespace-pre-wrap">{generatedText}</p>
                  ) : (
                    null
                  )}
                </div>
              </section>
            </section>
          </div>

          <section className="aethera-dock">
            <div className="aethera-sliders">
              {generationControls.map((control) => (
                <label key={control.key} className="aethera-slider-row">
                  <span className="aethera-slider-top">
                    <span>{control.label}</span>
                    <strong>{describeControlValue(control.key, settings[control.key])}</strong>
                  </span>
                  <input
                    type="range"
                    min={control.min}
                    max={control.max}
                    step={control.step}
                    value={settings[control.key]}
                    onChange={handleSettingChange(control.key)}
                    className="aethera-range"
                  />
                </label>
              ))}
            </div>

            <div className="aethera-actions aethera-actions-right">
              <button type="button" onClick={handleResetSettings} className="aethera-ghost">
                <RotateCcw size={15} />
                Reset settings
              </button>
              <a href={HUGGING_FACE_URL} target="_blank" rel="noreferrer" className="aethera-ghost">
                Hugging Face
              </a>
              <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="aethera-ghost">
                <Github size={15} />
                Repo
              </a>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
