import { useEffect, useMemo, useState } from 'react'
import '../App.css'

/* ================= 常量与类型 ================= */

const BUPT_BLUE = '#003399'
const ACCESS_CODE = 'dog'
const AUTH_KEY = 'dg-auth'
const KEY_STORAGE = 'dg-api-key'
const MODEL_STORAGE = 'dg-model'
const DEFAULT_MODEL = 'qwen3.7-max'
const ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'

interface DialogueTurn {
  speaker: string
  content: string
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

/* ================= API 调用 ================= */

function buildPrompt(form: FormState, variantIndex: number): string {
  return `请生成一段用于心理学共情实验的虚拟人物对话，要求如下：
- 对话主题：${form.topic}
- 对话轮数：${form.rounds} 轮（每轮包含角色A和角色B各发言一次，共 ${form.rounds * 2} 条发言）
- 角色A的性格特点：${form.personaA}
- 角色B的性格特点：${form.personaB}
- 对话风格：${form.style}
- 备注（实验人员的其他要求）：${form.notes || '无'}

要求：
1. 对话必须紧密围绕主题展开，自然流畅，严格符合双方的性格设定和对话风格。
2. 由角色A先发言，之后双方交替进行。
3. 每次发言长度适中（1~3句话），要有真实的情感表达，便于被试产生共情反应。
4. 这是第 ${variantIndex} 个候选版本（随机种子：${Math.floor(Math.random() * 100000)}），请确保内容具有多样性。
5. 只输出JSON，不要输出任何其他内容，格式为：
{"dialogue": [{"speaker": "角色A", "content": "..."}, {"speaker": "角色B", "content": "..."}, ...]}`
}

function parseDialogue(raw: string): DialogueTurn[] {
  let text = raw.trim()
  // 去掉可能的 markdown 代码块包裹
  text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')
  const start = text.indexOf('{')
  const end = text.lastIndexOf('}')
  if (start === -1 || end === -1) throw new Error('模型未返回有效的 JSON')
  const obj = JSON.parse(text.slice(start, end + 1))
  const arr = Array.isArray(obj) ? obj : obj.dialogue ?? obj.turns ?? obj.dialog
  if (!Array.isArray(arr) || arr.length === 0) throw new Error('JSON 中没有对话内容')
  return arr.map((t: { speaker?: string; content?: string; role?: string; text?: string }, i: number) => ({
    speaker: t.speaker || t.role || (i % 2 === 0 ? '角色A' : '角色B'),
    content: String(t.content ?? t.text ?? ''),
  }))
}

async function callQwen(apiKey: string, model: string, form: FormState, variantIndex: number): Promise<DialogueTurn[]> {
  const resp = await fetch(ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'system',
          content:
            '你是一位心理学实验对话设计师，专门为共情研究实验编写高质量的虚拟人物对话脚本。你严格遵守输出格式要求。',
        },
        { role: 'user', content: buildPrompt(form, variantIndex) },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.9,
    }),
  })
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '')
    throw new Error(`API 请求失败（HTTP ${resp.status}）：${detail.slice(0, 200)}`)
  }
  const data = await resp.json()
  const content = data?.choices?.[0]?.message?.content
  if (!content) throw new Error('模型返回内容为空')
  return parseDialogue(content)
}

/* ================= CSV 导出 ================= */

function csvEscape(v: string | number): string {
  const s = String(v)
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

function exportCSV(dialogues: Dialogue[], filename: string) {
  const header = ['dialogue_id', 'topic', 'rounds', 'persona_a', 'persona_b', 'style', 'notes', 'turn_index', 'speaker', 'content']
  const rows = dialogues.flatMap((d) =>
    d.turns.map((t, i) => [d.id, d.topic, d.rounds, d.personaA, d.personaB, d.style, d.notes, i + 1, t.speaker, t.content] as const),
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
  const [model, setModel] = useState(() => localStorage.getItem(MODEL_STORAGE) || DEFAULT_MODEL)
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
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [errors, setErrors] = useState<string[]>([])

  useEffect(() => {
    localStorage.setItem(KEY_STORAGE, apiKey)
  }, [apiKey])
  useEffect(() => {
    localStorage.setItem(MODEL_STORAGE, model)
  }, [model])

  const selectedDialogues = useMemo(() => dialogues.filter((d) => d.selected), [dialogues])

  if (!authed) return <LoginGate onPass={() => setAuthed(true)} />

  const update = <K extends keyof FormState>(key: K, value: FormState[K]) => setForm((f) => ({ ...f, [key]: value }))

  const validate = (): string | null => {
    if (!apiKey.trim()) return '请先在右上角「设置」中填写通义千问 API Key'
    if (!form.topic.trim()) return '请填写对话主题'
    if (!form.personaA.trim() || !form.personaB.trim()) return '请填写双方的性格特点'
    if (!form.style.trim()) return '请填写对话风格'
    if (form.rounds < 1 || form.rounds > 50) return '对话轮数需在 1~50 之间'
    if (form.batch < 1 || form.batch > 10) return '批量生成数量需在 1~10 之间'
    return null
  }

  const generate = async () => {
    const err = validate()
    if (err) {
      setErrors([err])
      return
    }
    setGenerating(true)
    setErrors([])
    setProgress({ done: 0, total: form.batch })
    const results = await Promise.allSettled(
      Array.from({ length: form.batch }, (_, i) =>
        callQwen(apiKey.trim(), model.trim() || DEFAULT_MODEL, form, i + 1).then((turns) => {
          setProgress((p) => ({ ...p, done: p.done + 1 }))
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
        })
      } else {
        newErrors.push(`第 ${i + 1} 段生成失败：${r.reason?.message || r.reason}`)
      }
    })
    setDialogues((prev) => [...newDialogues, ...prev])
    setErrors(newErrors)
    setGenerating(false)
  }

  const toggle = (id: number) =>
    setDialogues((prev) => prev.map((d) => (d.id === id ? { ...d, selected: !d.selected } : d)))
  const remove = (id: number) => setDialogues((prev) => prev.filter((d) => d.id !== id))
  const selectAll = (v: boolean) => setDialogues((prev) => prev.map((d) => ({ ...d, selected: v })))

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
            <p className="text-xs text-blue-200 mt-0.5">基于通义千问的虚拟人物对话素材构建工具 · 供心理学共情实验使用</p>
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
            <h2 className="font-semibold text-gray-800 mb-4">模型设置</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
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
                <label className={labelCls}>模型名称</label>
                <input type="text" value={model} onChange={(e) => setModel(e.target.value)} className={inputCls} />
                <p className="text-xs text-gray-400 mt-1">默认 {DEFAULT_MODEL}，调用阿里云百炼 DashScope 兼容接口</p>
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

          <div className="mt-5 flex items-center gap-4">
            <button
              onClick={generate}
              disabled={generating}
              className="text-white font-medium rounded-lg px-8 py-2.5 transition-opacity hover:opacity-90 disabled:opacity-50"
              style={{ backgroundColor: BUPT_BLUE }}
            >
              {generating ? `生成中… ${progress.done}/${progress.total}` : '批量生成对话'}
            </button>
            {generating && <span className="text-sm text-gray-500">正在调用大模型，请稍候（已并发生成多段对话）</span>}
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
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="font-semibold text-gray-800 mr-auto">
                候选对话（共 {dialogues.length} 段，已选 {selectedDialogues.length} 段）
              </h2>
              <button onClick={() => selectAll(true)} className="text-sm text-[#003399] hover:underline">
                全选
              </button>
              <button onClick={() => selectAll(false)} className="text-sm text-gray-500 hover:underline">
                取消全选
              </button>
              <button
                onClick={() => exportCSV(selectedDialogues, `dialogues_selected_${stamp()}.csv`)}
                disabled={selectedDialogues.length === 0}
                className="text-sm text-white rounded-lg px-4 py-1.5 disabled:opacity-40 hover:opacity-90"
                style={{ backgroundColor: BUPT_BLUE }}
              >
                导出选中 CSV
              </button>
              <button
                onClick={() => exportCSV(dialogues, `dialogues_all_${stamp()}.csv`)}
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

            {dialogues.map((d) => (
              <article
                key={d.id}
                className={`bg-white rounded-xl shadow-sm border p-5 transition-colors ${
                  d.selected ? 'border-[#003399] ring-1 ring-[#003399]' : 'border-gray-200'
                }`}
              >
                <div className="flex items-start gap-3 mb-3">
                  <input
                    type="checkbox"
                    checked={d.selected}
                    onChange={() => toggle(d.id)}
                    className="mt-1 h-4 w-4 accent-[#003399]"
                  />
                  <div className="mr-auto">
                    <span className="font-medium text-gray-800">对话 #{d.id}</span>
                    <span className="text-sm text-gray-500 ml-3">
                      主题：{d.topic} · 设定 {d.rounds} 轮 · 实际 {Math.floor(d.turns.length / 2)} 轮
                    </span>
                  </div>
                  <button
                    onClick={() => exportCSV([d], `dialogue_${d.id}_${stamp()}.csv`)}
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
                          <div className={`text-xs mb-1 ${isA ? 'text-[#003399]' : 'text-blue-200'}`}>{t.speaker}</div>
                          {t.content}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </article>
            ))}
          </section>
        )}

        {dialogues.length === 0 && !generating && (
          <section className="bg-white rounded-xl border border-dashed border-gray-300 p-12 text-center text-gray-400 text-sm">
            填写上方实验参数并点击「批量生成对话」，生成的候选对话将显示在这里，可勾选后导出为 CSV
          </section>
        )}

        <footer className="text-center text-xs text-gray-400 pt-2 pb-6">
          共情对话生成平台 · 仅供学术研究使用
        </footer>
      </main>
    </div>
  )
}
