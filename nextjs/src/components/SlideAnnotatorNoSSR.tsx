"use client";

import dynamic from "next/dynamic";

export const SlideAnnotatorNoSSR = dynamic(
  () => import("@/components/SlideAnnotator").then((mod) => mod.SlideAnnotator),
  {
    ssr: false,
  }
);
