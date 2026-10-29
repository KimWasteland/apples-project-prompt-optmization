"use client";

import { useState } from "react";

type Result = {
  prompt: string;
  label: { role: string; purpose: string; task: string; format: string; constraint: string };
  fixes: string[];
  has_content: boolean;
  latency: { total: number };
};

const TASK_NAME: Record<string, string> = {
  summarization: "요약",
  comparison: "비교",
  explanation: "설명",
  recommendation: "추천",
  planning: "기획",
};

const EXAMPLES = [
  "갤럭시 S25랑 아이폰 17 중에 사진 많이 찍는 사람한테 뭐가 나아? 결론만",
  "자취생 첫 전자레인지 추천해줘 20만원 이하로",
  "양자컴퓨터가 뭔지 초등학생도 알게 설명해줘",
  "우리 동아리 신입생 환영회 아이디어 좀 짜줘. 예산은 30만원이야",
];

export default function Home() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [copied, setCopied] = useState(false);

  async function submit() {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError("");
    setCopied(false);
    try {
      const res = await fetch("/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });
      if (!res.ok) throw new Error(`모델 서버 오류 (${res.status})`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof TypeError ? "모델 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요." : String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  }

  async function copy() {
    if (!result) return;
    await navigator.clipboard.writeText(result.prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-10 sm:py-16">
      <header className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">프롬프트 최적화</h1>
        <p className="mt-2 text-sm text-zinc-500">
          짧은 질문을 입력하면 역할, 출력 형식, 제약 조건이 담긴 프롬프트로 바꿔 드립니다. 요약할 글은 질문 아래에
          붙여 넣으세요.
        </p>
      </header>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit();
          }}
          rows={5}
          maxLength={20000}
          placeholder="예) 갤럭시 S25랑 아이폰 17 중에 뭐가 나아? 결론만"
          className="w-full resize-y bg-transparent text-base outline-none placeholder:text-zinc-400"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => setText(ex)}
              className="max-w-full truncate rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
            >
              {ex}
            </button>
          ))}
        </div>

        <div className="mt-4 flex justify-end border-t border-zinc-100 pt-4 dark:border-zinc-800">
          <button
            onClick={submit}
            disabled={loading || !text.trim()}
            className="h-11 shrink-0 rounded-xl bg-indigo-600 px-6 font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "생성 중…" : "최적화"}
          </button>
        </div>
      </section>

      {error && (
        <p className="mt-4 rounded-xl bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">
          {error}
        </p>
      )}

      {result && (
        <section className="mt-6 space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Chip strong>{TASK_NAME[result.label.task] ?? result.label.task}</Chip>
            <Chip>{result.label.role}</Chip>
            <Chip mono>{result.label.format}</Chip>
            <Chip mono>{result.label.constraint}</Chip>
            <span className="ml-auto text-zinc-400">{result.latency.total.toFixed(1)}초</span>
          </div>
          <div className="relative rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <button
              onClick={copy}
              className="absolute right-3 top-3 rounded-lg bg-zinc-100 px-3 py-1 text-xs font-medium hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700"
            >
              {copied ? "복사됨" : "복사"}
            </button>
            <pre className="whitespace-pre-wrap break-words p-5 pr-20 text-sm leading-relaxed">{result.prompt}</pre>
          </div>
        </section>
      )}
    </main>
  );
}

function Chip({ children, strong, mono }: { children: React.ReactNode; strong?: boolean; mono?: boolean }) {
  return (
    <span
      className={`rounded-full px-2.5 py-1 ${mono ? "font-mono" : ""} ${
        strong
          ? "bg-indigo-600 text-white"
          : "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
      }`}
    >
      {children}
    </span>
  );
}
