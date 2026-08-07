import { useEffect, useMemo, useState } from 'react'
import '../App.css'

/* ================= 常量与类型 ================= */

const BUPT_BLUE = '#003399'
const ACCESS_CODE = 'dog'
const AUTH_KEY = 'dg-auth'
const KEY_STORAGE = 'dg-api-key'
const MODEL_A_STORAGE = 'dg-model-agent1'
const MODEL_B_STORAGE = 'dg-model-agent2'
const REVIEWER2_STORAGE = 'dg-model-reviewer2'
const DUAL_STORAGE = 'dg-dual-review'
const DIVERGENCE_STORAGE = 'dg-divergence'
const THRESHOLD_STORAGE = 'dg-threshold'
const AUTOSCORE_STORAGE = 'dg-autoscore'
const DEFAULT_MODEL_A = 'qwen3.8-max'
const DEFAULT_MODEL_B = 'glm-5.2'
const DEFAULT_REVIEWER2 = 'deepseek-v4-flash-0731'
const ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
const QWEN_MODELS = ['qwen3.7-max', 'qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen-long']
const SCORE_DIMS = ['主题契合度', '角色性格一致性', '对话风格符合度', '共情激发潜力', '结构与轮数合规性']

interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface DialogueTurn {
  speaker: string
  content: string
}

interface DimensionScore {
  name: string
  score: number
  evidence: string
}

interface ReviewerReport {
  model: string
  dimensions: DimensionScore[]
  total: number
  comment: string
}

interface DialogueScore {
  dimensions: DimensionScore[]
  total: number
  maxTotal: number
  comment: string
  passed: boolean
  model: string
  /** 双评审模式下的各评审原始报告 */
  reviewers?: ReviewerReport[]
  /** 双评审总分差 */
  divergence?: number
  /** 分差过大，建议人工复核 */
  needsReview?: boolean
}

interface Dialogue {
  id: number
  topic: string
  rounds: number
  personaA: string
  personaB: string
  style: string
  notes: string
  turns: DialogueTurn[]
  selected: boolean
  score: DialogueScore | null
}

interface FormState {
  rounds: number
  topic: string
  personaA: string
  personaB: string
  style: string
  notes: string
  batch: number
}

/* ================= 底层 API ================= */

async function chat(apiKey: string, model: string, messages: ChatMessage[], temperature: number): Promise<string> {
  const resp = await fetch(ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      response_format: { type: 'json_object' },
      temperature,
    }),
  })
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '')
    throw new Error(`API 请求失败（HTTP ${resp.status}）：${detail.slice(0, 200)}`)
  }
  const data = await resp.json()
  const content = data?.choices?.[0]?.message?.content
  if (!content) throw new Error('模型返回内容为空')
  return content
}

function extractJson(raw: string): unknown {
  let text = raw.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')
  const start = text.indexOf('{')
  const end = text.lastIndexOf('}')
  if (start === -1 || end === -1) throw new Error('模型未返回有效的 JSON')
  return JSON.parse(text.slice(start, end + 1))
}

/* ================= Agent 1：对话生成（带校验与 refine 重试） ================= */

function buildGeneratePrompt(form: FormState, variantIndex: number): string {
  return `请生成一段用于心理学共情实验的虚拟人物对话，要求如下：
- 对话主题：${form.topic}
- 对话轮数：${form.rounds} 轮（每轮包含角色A和角色B各发言一次，共 ${form.rounds * 2} 条发言）
- 角色A的性格特点：${form.personaA}
- 角色B的性格特点：${form.personaB}
- 对话风格：${form.style}
- 备注（实验人员的其他要求）：${form.notes || '无'}

要求：
1. 对话必须紧密围绕主题展开，自然流畅，严格符合双方的性格设定和对话风格。
2. 由角色A先发言，之后双方交替进行，发言总数必须为 ${form.rounds * 2} 条。
3. 每次发言长度适中（1~3句话），要有真实的情感表达，便于被试产生共情反应。
4. 这是第 ${variantIndex} 个候选版本（随机种子：${Math.floor(Math.random() * 100000)}），请确保内容具有多样性。
5. 只输出JSON，不要输出任何其他内容，格式为：
{"dialogue": [{"speaker": "角色A", "content": "..."}, {"speaker": "角色B", "content": "..."}, ...]}`
}

function parseDialogue(raw: string): DialogueTurn[] {
  const obj = extractJson(raw) as Record<string, unknown>
  const arr = (Array.isArray(obj) ? obj : (obj.dialogue ?? obj.turns ?? obj.dialog)) as
    | { speaker?: string; content?: string; role?: string; text?: string }[]
    | undefined
  if (!Array.isArray(arr) || arr.length === 0) throw new Error('JSON 中没有对话内容')
  return arr.map((t, i) => ({
    speaker: t.speaker || t.role || (i % 2 === 0 ? '角色A' : '角色B'),
    content: String(t.content ?? t.text ?? '').trim(),
  }))
}

/** 程序化校验：不依赖模型自觉，保证轮数与内容完整性 */
function validateTurns(turns: DialogueTurn[], rounds: number): string | null {
  const expected = rounds * 2
  if (turns.length < expected) return `发言条数 ${turns.length} 少于要求的 ${expected} 条（${rounds} 轮）`
  if (turns.length > expected + 2) return `发言条数 ${turns.length} 明显多于要求的 ${expected} 条`
  const emptyIdx = turns.findIndex((t) => !t.content)
  if (emptyIdx !== -1) return `第 ${emptyIdx + 1} 条发言内容为空`
  return null
}

/** 生成 + 校验 + 失败时把问题反馈给模型自动 refine（最多 3 次尝试） */
async function generateDialogue(
  apiKey: string,
  model: string,
  form: FormState,
  variantIndex: number,
): Promise<DialogueTurn[]> {
  const messages: ChatMessage[] = [
    {
      role: 'system',
      content:
        '你是一位心理学实验对话设计师，专门为共情研究实验编写高质量的虚拟人物对话脚本。你严格遵守实验需求和输出格式要求。',
    },
    { role: 'user', content: buildGeneratePrompt(form, variantIndex) },
  ]
  let lastProblem = ''
  for (let attempt = 1; attempt <= 3; attempt++) {
    if (lastProblem) {
      messages.push({
        role: 'user',
        content: `你上一次的输出存在问题：${lastProblem}。请针对该问题进行修正（refine），重新生成完整对话，仍然只输出规定格式的JSON。`,
      })
    }
    const content = await chat(apiKey, model, messages, 0.9)
    messages.push({ role: 'assistant', content })
    try {
      const turns = parseDialogue(content)
      const problem = validateTurns(turns, form.rounds)
      if (!problem) return turns
      lastProblem = problem
    } catch (e) {
      lastProblem = `输出无法解析为JSON（${e instanceof Error ? e.message : String(e)}）`
    }
  }
  throw new Error(`生成结果连续未通过校验：${lastProblem}`)
}

/* ================= Agent 2：评审打分（按维度 + evidence，支持双评审共识） ================= */

function buildScorePrompt(d: Dialogue): string {
  const transcript = d.turns.map((t, i) => `${i + 1}. ${t.speaker}：${t.content}`).join('\n')
  return `以下是一段为心理学共情实验生成的虚拟人物对话，以及实验人员的设计需求。请作为独立评审，对这段对话进行质量评估。

【实验需求】
- 对话主题：${d.topic}
- 轮数要求：${d.rounds} 轮（双方各发言一次为一轮，共 ${d.rounds * 2} 条发言）
- 角色A的性格特点：${d.personaA}
- 角色B的性格特点：${d.personaB}
- 对话风格：${d.style}
- 备注（实验人员的其他要求）：${d.notes || '无'}

【待评审对话】
${transcript}

请从以下 5 个维度逐条打分（每维度 0~10 的整数分），并且每条打分必须给出 evidence：引用对话中的具体发言（注明序号）作为评分依据，不允许空泛评价。
1. 主题契合度：对话是否始终围绕指定主题展开
2. 角色性格一致性：双方发言是否稳定符合各自的性格设定
3. 对话风格符合度：整体语气和风格是否符合指定要求
4. 共情激发潜力：情感表达是否真实，是否容易让被试产生共情反应
5. 结构与轮数合规性：发言条数、交替顺序、长度是否符合要求

只输出JSON，不要输出任何其他内容，格式为：
{"dimensions": [{"name": "维度名", "score": 0到10的整数, "evidence": "引用具体发言的评分依据"}, ...共5项], "comment": "总体评价（1~2句话，指出主要优点和问题）"}`
}

function parseReviewerReport(raw: string, model: string): ReviewerReport {
  const obj = extractJson(raw) as {
    dimensions?: { name?: string; score?: number; evidence?: string }[]
    comment?: string
  }
  if (!Array.isArray(obj.dimensions) || obj.dimensions.length === 0) {
    throw new Error('评分结果中没有维度数据')
  }
  // 分数钳制到 0~10 并取整，防止模型给出越界值
  const dimensions: DimensionScore[] = obj.dimensions.map((d, i) => ({
    name: d.name || SCORE_DIMS[i] || `维度${i + 1}`,
    score: Math.max(0, Math.min(10, Math.round(Number(d.score) || 0))),
    evidence: String(d.evidence ?? ''),
  }))
  // 总分由程序重新求和，不信任模型自报的总分（防止加总幻觉）
  const total = dimensions.reduce((s, d) => s + d.score, 0)
  return { model, dimensions, total, comment: String(obj.comment ?? '') }
}

async function reviewOnce(apiKey: string, model: string, d: Dialogue): Promise<ReviewerReport> {
  const raw = await chat(
    apiKey,
    model,
    [
      {
        role: 'system',
        content:
          '你是一名严格、公正的心理学实验数据质量评审专家。你的评分必须有对话原文作为依据，分数分布应有区分度，不要一律给高分。你严格遵守输出格式要求。',
      },
      { role: 'user', content: buildScorePrompt(d) },
    ],
    0.2, // 低温度，保证评分稳定可复现
  )
  return parseReviewerReport(raw, model)
}

/** 单评审 */
async function scoreDialogue(apiKey: string, model: string, d: Dialogue, threshold: number): Promise<DialogueScore> {
  const r = await reviewOnce(apiKey, model, d)
  return {
    dimensions: r.dimensions,
    total: r.total,
    maxTotal: r.dimensions.length * 10,
    comment: r.comment,
    passed: r.total >= threshold,
    model,
  }
}

/** 双评审共识：两位独立评审，维度分取平均；总分差过大则标记需人工复核 */
async function scoreDialogueDual(
  apiKey: string,
  model1: string,
  model2: string,
  d: Dialogue,
  threshold: number,
  divergenceLimit: number,
): Promise<DialogueScore> {
  const [r1, r2] = await Promise.all([
    reviewOnce(apiKey, model1, d),
    reviewOnce(apiKey, model2, d),
  ])
  // 共识维度分：两评审取平均（四舍五入）；evidence 合并双方依据
  const dimensions: DimensionScore[] = r1.dimensions.map((dim, i) => {
    const other = r2.dimensions[i]
    return {
      name: dim.name,
      score: Math.round((dim.score + (other?.score ?? dim.score)) / 2),
      evidence: `[${model1}] ${dim.evidence || '（无）'} ｜ [${model2}] ${other?.evidence || '（无）'}`,
    }
  })
  const total = dimensions.reduce((s, x) => s + x.score, 0)
  const divergence = Math.abs(r1.total - r2.total)
  return {
    dimensions,
    total,
    maxTotal: dimensions.length * 10,
    comment: r2.comment ? `${r1.comment} ｜ ${r2.comment}` : r1.comment,
    passed: total >= threshold,
    model: `${model1} + ${model2}`,
    reviewers: [r1, r2],
    divergence,
    needsReview: divergence > divergenceLimit,
  }
}

/* ================= CSV 导出 ================= */

function csvEscape(v: string | number): string {
  const s = String(v)
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

function scoreSummary(s: DialogueScore): string {
  return s.dimensions.map((d) => `${d.name}=${d.score}（依据：${d.evidence}）`).join('；')
}

function exportCSV(dialogues: Dialogue[], ranks: Map<number, number>, filename: string) {
  const header = [
    'dialogue_id', 'rank', 'topic', 'rounds', 'persona_a', 'persona_b', 'style', 'notes',
    'turn_index', 'speaker', 'content',
    'score_total', 'score_max', 'score_passed', 'needs_review', 'divergence',
    'reviewer1', 'reviewer1_total', 'reviewer2', 'reviewer2_total',
    'score_dimensions', 'score_comment', 'scored_by',
  ]
  const rows = dialogues.flatMap((d) =>
    d.turns.map((t, i) => [
      d.id,
      d.score ? (ranks.get(d.id) ?? '') : '',
      d.topic, d.rounds, d.personaA, d.personaB, d.style, d.notes,
      i + 1, t.speaker, t.content,
      d.score?.total ?? '', d.score?.maxTotal ?? '',
      d.score ? (d.score.passed ? 'PASS' : 'FAIL') : '',
      d.score?.needsReview ? 'YES' : '',
      d.score?.divergence ?? '',
      d.score?.reviewers?.[0]?.model ?? '', d.score?.reviewers?.[0]?.total ?? '',
      d.score?.reviewers?.[1]?.model ?? '', d.score?.reviewers?.[1]?.total ?? '',
      d.score ? scoreSummary(d.score) : '',
      d.score?.comment ?? '', d.score?.model ?? '',
    ] as const),
  )
  const csv = '﻿' + [header, ...rows].map((r) => r.map(csvEscape).join(',')).join('\r\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

/* ================= 登录页 ================= */

function LoginGate({ onPass }: { onPass: () => void }) {
  const [value, setValue] = useState('')
  const [error, setError] = useState(false)

  const submit = () => {
    if (value.trim().toLowerCase() === ACCESS_CODE) {
      sessionStorage.setItem(AUTH_KEY, '1')
      onPass()
    } else {
      setError(true)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: BUPT_BLUE }}>
      <div className="bg-white rounded-xl shadow-2xl p-10 w-full max-w-sm">
        <h1 className="text-2xl font-bold text-center" style={{ color: BUPT_BLUE }}>
          共情对话生成平台
        </h1>
        <p className="text-sm text-gray-500 text-center mt-2 mb-6">请输入访问口令进入系统</p>
        <input
          type="password"
          value={value}
          onChange={(e) => {
            setValue(e.target.value)
            setError(false)
          }}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          placeholder="访问口令"
          className="w-full border border-gray-300 rounded-lg px-4 py-2.5 outline-none focus:ring-2 focus:ring-[#003399] focus:border-transparent"
        />
        {error && <p className="text-red-500 text-sm mt-2">口令错误，请重试</p>}
        <button
          onClick={submit}
          className="w-full mt-4 text-white font-medium rounded-lg py-2.5 transition-opacity hover:opacity-90"
          style={{ backgroundColor: BUPT_BLUE }}
        >
          进入
        </button>
      </div>
    </div>
  )
}

/* ================= 主页面 ================= */

let nextId = 1

export default function Home() {
  const [authed, setAuthed] = useState(() => sessionStorage.getItem(AUTH_KEY) === '1')
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(KEY_STORAGE) || '')
  const [agent1Model, setAgent1Model] = useState(
    () => localStorage.getItem(MODEL_A_STORAGE) || localStorage.getItem('dg-model') || DEFAULT_MODEL_A,
  )
  const [agent2Model, setAgent2Model] = useState(() => localStorage.getItem(MODEL_B_STORAGE) || DEFAULT_MODEL_B)
  const [reviewer2Model, setReviewer2Model] = useState(() => localStorage.getItem(REVIEWER2_STORAGE) || DEFAULT_REVIEWER2)
  const [dualReview, setDualReview] = useState(() => localStorage.getItem(DUAL_STORAGE) === '1')
  const [divergenceLimit, setDivergenceLimit] = useState(() => Number(localStorage.getItem(DIVERGENCE_STORAGE)) || 8)
  const [threshold, setThreshold] = useState(() => Number(localStorage.getItem(THRESHOLD_STORAGE)) || 35)
  const [autoScore, setAutoScore] = useState(() => localStorage.getItem(AUTOSCORE_STORAGE) !== '0')
  const [showSettings, setShowSettings] = useState(false)

  const [form, setForm] = useState<FormState>({
    rounds: 6,
    topic: '',
    personaA: '',
    personaB: '',
    style: '',
    notes: '',
    batch: 3,
  })
  const [dialogues, setDialogues] = useState<Dialogue[]>([])
  const [generating, setGenerating] = useState(false)
  const [genProgress, setGenProgress] = useState({ done: 0, total: 0 })
  const [scoring, setScoring] = useState(false)
  const [scoreProgress, setScoreProgress] = useState({ done: 0, total: 0 })
  const [errors, setErrors] = useState<string[]>([])
  const [onlyPassed, setOnlyPassed] = useState(false)
  const [sortBy, setSortBy] = useState<'time' | 'score'>('time')

  useEffect(() => localStorage.setItem(KEY_STORAGE, apiKey), [apiKey])
  useEffect(() => localStorage.setItem(MODEL_A_STORAGE, agent1Model), [agent1Model])
  useEffect(() => localStorage.setItem(MODEL_B_STORAGE, agent2Model), [agent2Model])
  useEffect(() => localStorage.setItem(REVIEWER2_STORAGE, reviewer2Model), [reviewer2Model])
  useEffect(() => localStorage.setItem(DUAL_STORAGE, dualReview ? '1' : '0'), [dualReview])
  useEffect(() => localStorage.setItem(DIVERGENCE_STORAGE, String(divergenceLimit)), [divergenceLimit])
  useEffect(() => localStorage.setItem(THRESHOLD_STORAGE, String(threshold)), [threshold])
  useEffect(() => localStorage.setItem(AUTOSCORE_STORAGE, autoScore ? '1' : '0'), [autoScore])

  /** 排名：按总分降序，仅已评分的对话参与排名 */
  const ranks = useMemo(() => {
    const scored = dialogues.filter((d) => d.score).sort((a, b) => (b.score!.total - a.score!.total) || a.id - b.id)
    return new Map(scored.map((d, i) => [d.id, i + 1]))
  }, [dialogues])

  const displayList = useMemo(() => {
    let list = [...dialogues]
    if (onlyPassed) list = list.filter((d) => d.score?.passed)
    if (sortBy === 'score') {
      list.sort((a, b) => {
        if (a.score && b.score) return b.score.total - a.score.total || a.id - b.id
        if (a.score) return -1
        if (b.score) return 1
        return b.id - a.id
      })
    }
    return list
  }, [dialogues, onlyPassed, sortBy])

  const selectedDialogues = useMemo(() => dialogues.filter((d) => d.selected), [dialogues])
  const scoredCount = useMemo(() => dialogues.filter((d) => d.score).length, [dialogues])
  const passedCount = useMemo(() => dialogues.filter((d) => d.score?.passed).length, [dialogues])
  const reviewCount = useMemo(() => dialogues.filter((d) => d.score?.needsReview).length, [dialogues])

  if (!authed) return <LoginGate onPass={() => setAuthed(true)} />

  const update = <K extends keyof FormState>(key: K, value: FormState[K]) => setForm((f) => ({ ...f, [key]: value }))

  const validate = (): string | null => {
    if (!apiKey.trim()) return '请先在右上角「设置」中填写通义千问 API Key'
    if (!agent1Model.trim() || !agent2Model.trim()) return '请在「设置」中为 Agent1 和 Agent2 选择模型'
    if (dualReview) {
      if (!reviewer2Model.trim()) return '双评审模式下请为「评审模型 B」选择模型'
      if (agent2Model.trim() === reviewer2Model.trim()) return '双评审模式下两个评审模型不能相同，否则共识没有意义'
    }
    if (!form.topic.trim()) return '请填写对话主题'
    if (!form.personaA.trim() || !form.personaB.trim()) return '请填写双方的性格特点'
    if (!form.style.trim()) return '请填写对话风格'
    if (form.rounds < 1 || form.rounds > 50) return '对话轮数需在 1~50 之间'
    if (form.batch < 1 || form.batch > 10) return '批量生成数量需在 1~10 之间'
    return null
  }

  /** 对一段对话执行评分（按当前设置自动选择单/双评审） */
  const scoreOne = (d: Dialogue): Promise<DialogueScore> =>
    dualReview
      ? scoreDialogueDual(apiKey.trim(), agent2Model.trim(), reviewer2Model.trim(), d, threshold, divergenceLimit)
      : scoreDialogue(apiKey.trim(), agent2Model.trim(), d, threshold)

  /** 对一组对话执行 Agent2 评分（内部复用） */
  const runScoring = async (targets: Dialogue[]) => {
    if (targets.length === 0) return
    setScoring(true)
    setScoreProgress({ done: 0, total: targets.length })
    const scoreErrors: string[] = []
    const results = await Promise.allSettled(
      targets.map((d) =>
        scoreOne(d).then((s) => {
          setScoreProgress((p) => ({ ...p, done: p.done + 1 }))
          return { id: d.id, score: s }
        }),
      ),
    )
    const scoreMap = new Map<number, DialogueScore>()
    results.forEach((r) => {
      if (r.status === 'fulfilled') scoreMap.set(r.value.id, r.value.score)
      else scoreErrors.push(`评分失败：${r.reason?.message || r.reason}`)
    })
    // 写入评分；自动勾选「通过且无需人工复核」的对话
    setDialogues((prev) =>
      prev.map((d) => {
        const s = scoreMap.get(d.id)
        return s ? { ...d, score: s, selected: s.passed && !s.needsReview } : d
      }),
    )
    if (scoreErrors.length) setErrors((prev) => [...prev, ...scoreErrors])
    setScoring(false)
    setSortBy('score')
  }

  /** Agent1 批量生成（含校验 refine），可选自动进入 Agent2 评分 */
  const generate = async () => {
    const err = validate()
    if (err) {
      setErrors([err])
      return
    }
    setGenerating(true)
    setErrors([])
    setGenProgress({ done: 0, total: form.batch })
    const results = await Promise.allSettled(
      Array.from({ length: form.batch }, (_, i) =>
        generateDialogue(apiKey.trim(), agent1Model.trim(), form, i + 1).then((turns) => {
          setGenProgress((p) => ({ ...p, done: p.done + 1 }))
          return turns
        }),
      ),
    )
    const newDialogues: Dialogue[] = []
    const newErrors: string[] = []
    results.forEach((r, i) => {
      if (r.status === 'fulfilled') {
        newDialogues.push({
          id: nextId++,
          topic: form.topic,
          rounds: form.rounds,
          personaA: form.personaA,
          personaB: form.personaB,
          style: form.style,
          notes: form.notes,
          turns: r.value,
          selected: false,
          score: null,
        })
      } else {
        newErrors.push(`第 ${i + 1} 段生成失败：${r.reason?.message || r.reason}`)
      }
    })
    setDialogues((prev) => [...newDialogues, ...prev])
    setErrors(newErrors)
    setGenerating(false)
    // 生成后自动交给 Agent2 评分与预筛选
    if (autoScore && newDialogues.length > 0) {
      await runScoring(newDialogues)
    }
  }

  /** 手动触发：对所有未评分对话执行 Agent2 评分 */
  const scoreAll = async () => {
    if (!apiKey.trim()) {
      setErrors(['请先在「设置」中填写 API Key'])
      return
    }
    await runScoring(dialogues.filter((d) => !d.score))
  }

  /** 单段重新生成（Agent1 refine） */
  const regenerate = async (d: Dialogue) => {
    setDialogues((prev) => prev.map((x) => (x.id === d.id ? { ...x, score: null } : x)))
    try {
      const turns = await generateDialogue(
        apiKey.trim(),
        agent1Model.trim(),
        { rounds: d.rounds, topic: d.topic, personaA: d.personaA, personaB: d.personaB, style: d.style, notes: d.notes, batch: 1 },
        Math.floor(Math.random() * 1000) + 100,
      )
      setDialogues((prev) => prev.map((x) => (x.id === d.id ? { ...x, turns, selected: false } : x)))
    } catch (e) {
      setErrors([`对话 #${d.id} 重新生成失败：${e instanceof Error ? e.message : String(e)}`])
    }
  }

  const toggle = (id: number) =>
    setDialogues((prev) => prev.map((d) => (d.id === id ? { ...d, selected: !d.selected } : d)))
  const remove = (id: number) => setDialogues((prev) => prev.filter((d) => d.id !== id))
  const selectAll = (v: boolean) => setDialogues((prev) => prev.map((d) => ({ ...d, selected: v })))
  const selectPassed = () =>
    setDialogues((prev) => prev.map((d) => ({ ...d, selected: !!d.score?.passed && !d.score?.needsReview })))

  const stamp = () => {
    const d = new Date()
    const p = (n: number) => String(n).padStart(2, '0')
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`
  }

  const inputCls =
    'w-full border border-gray-300 rounded-lg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[#003399] focus:border-transparent'
  const labelCls = 'block text-sm font-medium text-gray-700 mb-1'

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 顶栏 */}
      <header className="text-white shadow" style={{ backgroundColor: BUPT_BLUE }}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">共情对话生成平台</h1>
            <p className="text-xs text-blue-200 mt-0.5">
              双 Agent 流水线：Agent1 生成 → Agent2 评分排名 → 研究人员筛选导出
            </p>
          </div>
          <button
            onClick={() => setShowSettings((v) => !v)}
            className="text-sm border border-blue-300 rounded-lg px-4 py-1.5 hover:bg-white/10 transition-colors"
          >
            设置
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6 space-y-6">
        {/* 设置面板 */}
        {showSettings && (
          <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="font-semibold text-gray-800 mb-4">模型与流水线设置</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="md:col-span-2">
                <label className={labelCls}>通义千问 API Key</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  className={inputCls}
                />
                <p className="text-xs text-gray-400 mt-1">仅保存在本地浏览器（localStorage），不会上传到任何服务器</p>
              </div>
              <div>
                <label className={labelCls}>Agent 1 · 生成模型</label>
                <input
                  type="text"
                  list="qwen-models"
                  value={agent1Model}
                  onChange={(e) => setAgent1Model(e.target.value)}
                  className={inputCls}
                />
                <p className="text-xs text-gray-400 mt-1">负责按要求生成对话（默认 {DEFAULT_MODEL_A}）</p>
              </div>
              <div>
                <label className={labelCls}>Agent 2 · 评审模型 A</label>
                <input
                  type="text"
                  list="qwen-models"
                  value={agent2Model}
                  onChange={(e) => setAgent2Model(e.target.value)}
                  className={inputCls}
                />
                <datalist id="qwen-models">
                  {QWEN_MODELS.map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
                <p className="text-xs text-gray-400 mt-1">
                  负责打分排名（默认 {DEFAULT_MODEL_B}，建议与 Agent1 不同以降低同源偏差）
                </p>
              </div>

              {/* 双评审共识 */}
              <div className="md:col-span-2 border border-gray-200 rounded-lg p-4 bg-gray-50/60">
                <label className="flex items-center gap-2 text-sm font-medium text-gray-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={dualReview}
                    onChange={(e) => setDualReview(e.target.checked)}
                    className="h-4 w-4 accent-[#003399]"
                  />
                  双评审共识模式（两个不同模型独立评分，取共识，分歧过大标记人工复核）
                </label>
                {dualReview && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                    <div>
                      <label className={labelCls}>评审模型 B</label>
                      <input
                        type="text"
                        list="qwen-models"
                        value={reviewer2Model}
                        onChange={(e) => setReviewer2Model(e.target.value)}
                        className={inputCls}
                      />
                      <p className="text-xs text-gray-400 mt-1">默认 {DEFAULT_REVIEWER2}，必须与评审模型 A 不同</p>
                    </div>
                    <div>
                      <label className={labelCls}>分歧阈值（总分差）</label>
                      <input
                        type="number"
                        min={1}
                        max={50}
                        value={divergenceLimit}
                        onChange={(e) => setDivergenceLimit(Number(e.target.value))}
                        className={inputCls}
                      />
                      <p className="text-xs text-gray-400 mt-1">
                        两位评审总分差 &gt; 该值时标记「需人工复核」，不自动勾选
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <div>
                <label className={labelCls}>预筛选及格线（满分 50）</label>
                <input
                  type="number"
                  min={0}
                  max={50}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className={inputCls}
                />
                <p className="text-xs text-gray-400 mt-1">总分 ≥ 及格线的对话判定为 PASS 并自动勾选</p>
              </div>
              <div className="flex items-end pb-1">
                <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={autoScore}
                    onChange={(e) => setAutoScore(e.target.checked)}
                    className="h-4 w-4 accent-[#003399]"
                  />
                  生成后自动评分与预筛选
                </label>
              </div>
            </div>
          </section>
        )}

        {/* 实验参数表单 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="font-semibold text-gray-800 mb-4">实验参数</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className={labelCls}>对话轮数 *</label>
              <input
                type="number"
                min={1}
                max={50}
                value={form.rounds}
                onChange={(e) => update('rounds', Number(e.target.value))}
                className={inputCls}
              />
              <p className="text-xs text-gray-400 mt-1">1 轮 = 双方各发言一次</p>
            </div>
            <div className="md:col-span-2">
              <label className={labelCls}>对话主题 *</label>
              <input
                type="text"
                value={form.topic}
                onChange={(e) => update('topic', e.target.value)}
                placeholder="例如：失业后向朋友倾诉"
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>角色A 性格特点 *</label>
              <textarea
                value={form.personaA}
                onChange={(e) => update('personaA', e.target.value)}
                placeholder="例如：内向、敏感，正经历挫折"
                rows={2}
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>角色B 性格特点 *</label>
              <textarea
                value={form.personaB}
                onChange={(e) => update('personaB', e.target.value)}
                placeholder="例如：外向、热情，善于倾听"
                rows={2}
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>对话风格 *</label>
              <textarea
                value={form.style}
                onChange={(e) => update('style', e.target.value)}
                placeholder="例如：情感支持型，语气温暖真诚"
                rows={2}
                className={inputCls}
              />
            </div>
            <div className="md:col-span-2">
              <label className={labelCls}>备注（实验人员的其他想法）</label>
              <textarea
                value={form.notes}
                onChange={(e) => update('notes', e.target.value)}
                placeholder="例如：希望角色B在对话中段表现出一次明显的共情回应；避免涉及具体地名"
                rows={2}
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>批量生成数量 *</label>
              <input
                type="number"
                min={1}
                max={10}
                value={form.batch}
                onChange={(e) => update('batch', Number(e.target.value))}
                className={inputCls}
              />
              <p className="text-xs text-gray-400 mt-1">一次生成多段候选对话（1~10）</p>
            </div>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-4">
            <button
              onClick={generate}
              disabled={generating || scoring}
              className="text-white font-medium rounded-lg px-8 py-2.5 transition-opacity hover:opacity-90 disabled:opacity-50"
              style={{ backgroundColor: BUPT_BLUE }}
            >
              {generating
                ? `Agent1 生成中… ${genProgress.done}/${genProgress.total}`
                : scoring
                  ? `Agent2 评分中… ${scoreProgress.done}/${scoreProgress.total}`
                  : autoScore
                    ? '生成并自动评分'
                    : '批量生成对话'}
            </button>
            {dialogues.some((d) => !d.score) && !generating && (
              <button
                onClick={scoreAll}
                disabled={scoring}
                className="text-sm border border-[#003399] text-[#003399] rounded-lg px-4 py-2 hover:bg-blue-50 disabled:opacity-40"
              >
                {scoring ? `评分中… ${scoreProgress.done}/${scoreProgress.total}` : '对未评分对话执行评分'}
              </button>
            )}
            {(generating || scoring) && (
              <span className="text-sm text-gray-500">正在调用大模型，请稍候</span>
            )}
          </div>

          {errors.length > 0 && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 space-y-1">
              {errors.map((e, i) => (
                <p key={i} className="text-sm text-red-600">
                  {e}
                </p>
              ))}
            </div>
          )}
        </section>

        {/* 结果区 */}
        {dialogues.length > 0 && (
          <section className="space-y-4">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <h2 className="font-semibold text-gray-800 mr-auto">
                候选对话（共 {dialogues.length} 段 · 已评分 {scoredCount} · 通过 {passedCount}
                {reviewCount > 0 && <span className="text-amber-600"> · 需复核 {reviewCount}</span>} · 已选{' '}
                {selectedDialogues.length}）
              </h2>
              <button onClick={() => selectAll(true)} className="text-sm text-[#003399] hover:underline">
                全选
              </button>
              <button onClick={() => selectAll(false)} className="text-sm text-gray-500 hover:underline">
                取消全选
              </button>
              <button onClick={selectPassed} className="text-sm text-[#003399] hover:underline">
                仅选通过
              </button>
              <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={onlyPassed}
                  onChange={(e) => setOnlyPassed(e.target.checked)}
                  className="h-4 w-4 accent-[#003399]"
                />
                仅显示通过
              </label>
              <button
                onClick={() => setSortBy((s) => (s === 'score' ? 'time' : 'score'))}
                className="text-sm border border-gray-300 text-gray-600 rounded-lg px-3 py-1.5 hover:bg-gray-100"
              >
                {sortBy === 'score' ? '按得分排名 ↓' : '按生成时间'}
              </button>
              <button
                onClick={() => exportCSV(selectedDialogues, ranks, `dialogues_selected_${stamp()}.csv`)}
                disabled={selectedDialogues.length === 0}
                className="text-sm text-white rounded-lg px-4 py-1.5 disabled:opacity-40 hover:opacity-90"
                style={{ backgroundColor: BUPT_BLUE }}
              >
                导出选中 CSV
              </button>
              <button
                onClick={() => exportCSV(dialogues, ranks, `dialogues_all_${stamp()}.csv`)}
                className="text-sm border border-[#003399] text-[#003399] rounded-lg px-4 py-1.5 hover:bg-blue-50"
              >
                导出全部 CSV
              </button>
              <button
                onClick={() => setDialogues([])}
                className="text-sm border border-gray-300 text-gray-500 rounded-lg px-4 py-1.5 hover:bg-gray-100"
              >
                清空
              </button>
            </div>

            {displayList.map((d) => (
              <article
                key={d.id}
                className={`bg-white rounded-xl shadow-sm border p-5 transition-colors ${
                  d.selected ? 'border-[#003399] ring-1 ring-[#003399]' : 'border-gray-200'
                }`}
              >
                <div className="flex flex-wrap items-center gap-3 mb-3">
                  <input
                    type="checkbox"
                    checked={d.selected}
                    onChange={() => toggle(d.id)}
                    className="h-4 w-4 accent-[#003399]"
                  />
                  <div className="mr-auto flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-gray-800">对话 #{d.id}</span>
                    {d.score && (
                      <>
                        <span
                          className="text-xs font-semibold text-white rounded-full px-2.5 py-0.5"
                          style={{ backgroundColor: d.score.passed ? '#15803d' : '#b91c1c' }}
                        >
                          {d.score.passed ? 'PASS' : 'FAIL'} {d.score.total}/{d.score.maxTotal}
                        </span>
                        {d.score.needsReview && (
                          <span className="text-xs font-semibold text-amber-700 bg-amber-100 rounded-full px-2.5 py-0.5">
                            需人工复核（分差 {d.score.divergence}）
                          </span>
                        )}
                        <span className="text-xs text-gray-500">排名 #{ranks.get(d.id)}</span>
                        <span className="text-xs text-gray-400">评审：{d.score.model}</span>
                      </>
                    )}
                    {!d.score && <span className="text-xs text-gray-400">未评分</span>}
                    <span className="text-sm text-gray-500">
                      主题：{d.topic} · 设定 {d.rounds} 轮 · 实际 {Math.floor(d.turns.length / 2)} 轮
                    </span>
                  </div>
                  <button
                    onClick={() => regenerate(d)}
                    disabled={generating || scoring}
                    className="text-xs text-[#003399] hover:underline disabled:opacity-40"
                  >
                    重新生成
                  </button>
                  <button
                    onClick={() => exportCSV([d], ranks, `dialogue_${d.id}_${stamp()}.csv`)}
                    className="text-xs text-[#003399] hover:underline"
                  >
                    导出此段
                  </button>
                  <button onClick={() => remove(d.id)} className="text-xs text-red-500 hover:underline">
                    删除
                  </button>
                </div>

                <div className="space-y-2.5">
                  {d.turns.map((t, i) => {
                    const isA = /A/i.test(t.speaker) || i % 2 === 0
                    return (
                      <div key={i} className={`flex ${isA ? 'justify-start' : 'justify-end'}`}>
                        <div
                          className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                            isA ? 'bg-blue-50 text-gray-800 rounded-tl-sm' : 'text-white rounded-tr-sm'
                          }`}
                          style={isA ? undefined : { backgroundColor: BUPT_BLUE }}
                        >
                          <div className={`text-xs mb-1 ${isA ? 'text-[#003399]' : 'text-blue-200'}`}>
                            {i + 1}. {t.speaker}
                          </div>
                          {t.content}
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Agent2 评审报告 */}
                {d.score && (
                  <div className="mt-4 border-t border-gray-100 pt-4">
                    <div className="text-xs font-semibold text-gray-500 mb-2">
                      Agent2 评审报告{d.score.reviewers && '（双评审共识）'}
                    </div>
                    <div className="space-y-2">
                      {d.score.dimensions.map((dim, i) => (
                        <div key={i} className="text-xs">
                          <div className="flex items-center gap-2">
                            <span className="w-28 shrink-0 text-gray-600">{dim.name}</span>
                            <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{ width: `${dim.score * 10}%`, backgroundColor: BUPT_BLUE }}
                              />
                            </div>
                            <span className="w-8 text-right font-medium text-gray-700">{dim.score}</span>
                          </div>
                          {dim.evidence && (
                            <p className="text-gray-400 mt-0.5 md:ml-[7.5rem]">依据：{dim.evidence}</p>
                          )}
                        </div>
                      ))}
                    </div>
                    {d.score.reviewers && (
                      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3">
                        {d.score.reviewers.map((r, i) => (
                          <span key={i} className="text-xs text-gray-500 bg-gray-50 rounded-lg px-3 py-1.5">
                            评审{i + 1}（{r.model}）：{r.total}/{d.score!.maxTotal} 分 — {r.comment}
                          </span>
                        ))}
                      </div>
                    )}
                    {d.score.comment && (
                      <p className="text-xs text-gray-500 mt-2 bg-gray-50 rounded-lg px-3 py-2">
                        总评：{d.score.comment}
                      </p>
                    )}
                  </div>
                )}
              </article>
            ))}
          </section>
        )}

        {dialogues.length === 0 && !generating && (
          <section className="bg-white rounded-xl border border-dashed border-gray-300 p-12 text-center text-gray-400 text-sm">
            填写上方实验参数并点击生成，Agent1 产出候选对话后由 Agent2 按维度打分排名，通过预筛选的对话将自动勾选，可一键导出
            CSV
          </section>
        )}

        <footer className="text-center text-xs text-gray-400 pt-2 pb-6">
          共情对话生成平台 · 双 Agent 可控生成流水线 · 仅供学术研究使用
        </footer>
      </main>
    </div>
  )
}
