import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export", // 정적 파일(out/)로 빌드 -> FastAPI가 서빙
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
