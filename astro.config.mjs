import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";
import icon from "astro-icon";
import mdx from "@astrojs/mdx";

import react from "@astrojs/react";

import svelte from "@astrojs/svelte";

// https://astro.build/config
export default defineConfig({
  site: 'https://yourusername.github.io', // 替换为你的 GitHub 用户名  
  base: '/website_source', // 替换为你的仓库名
  integrations: [tailwind(), icon(), mdx(), react(), svelte()],
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    }
  }
});