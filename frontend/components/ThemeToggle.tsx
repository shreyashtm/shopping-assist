"use client";

import { useSyncExternalStore } from "react";

type Theme = "dark" | "light";

/*
 * The applied palette lives on the document element, not in React state.
 *
 * The inline script in layout.tsx sets it before first paint, so a React copy
 * of the same fact would start out wrong on every load and need correcting in
 * an effect -- one render with the wrong label. Subscribing to the attribute
 * instead means there is exactly one source of truth, and the button reads it
 * rather than duplicating it.
 */
function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  return () => observer.disconnect();
}

function getSnapshot(): Theme {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

// Dark is the stylesheet default, so that is what the server renders.
function getServerSnapshot(): Theme {
  return "dark";
}

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const next: Theme = theme === "dark" ? "light" : "dark";

  function toggle() {
    if (next === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    try {
      localStorage.setItem("theme", next);
    } catch {
      // Private browsing denies localStorage; the toggle still works for this
      // session, it just will not be remembered.
    }
  }

  return (
    <button
      onClick={toggle}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className="border border-border px-3 py-1.5 text-xs text-muted transition-colors hover:border-foreground hover:text-foreground"
    >
      {next === "light" ? "Light" : "Dark"}
    </button>
  );
}
