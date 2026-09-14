import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // This repo already has its own root-level README/DECISIONS.md
  // documentation conventions; don't let `next dev` regenerate
  // AGENTS.md/CLAUDE.md inside frontend/ on every run.
  agentRules: false,
};

export default nextConfig;
