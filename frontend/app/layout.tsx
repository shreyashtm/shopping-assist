import type { Metadata } from "next";
import { Geist, Noticia_Text } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });

// The editorial serif, used for asides and statistics.
const noticia = Noticia_Text({
  variable: "--font-noticia",
  subsets: ["latin"],
  weight: ["400", "700"],
});

export const metadata: Metadata = {
  title: "Shopping Assistant",
  description:
    "Describe the occasion in plain English. Get a curated, explained shortlist of real products.",
};

/*
 * Applies the stored palette before first paint.
 *
 * Runs as a blocking inline script rather than in an effect: a effect-driven
 * toggle runs after hydration, which means one frame of the wrong palette on
 * every load. Dark is the stylesheet default, so only an explicit "light"
 * choice needs an attribute -- nothing to do in the common case.
 */
const THEME_SCRIPT = `
(function () {
  try {
    if (localStorage.getItem("theme") === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    }
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${noticia.variable} h-full`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
