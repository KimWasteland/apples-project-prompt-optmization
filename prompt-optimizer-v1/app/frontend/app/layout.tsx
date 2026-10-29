import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "프롬프트 최적화",
  description: "짧은 질문을 역할·형식·제약 조건이 담긴 최적화 프롬프트로 바꿔 줍니다.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ko" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
