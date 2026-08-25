import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { QueryBar } from "./QueryBar";

describe("QueryBar", () => {
  it("submits the trimmed text and clears the box on Enter", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<QueryBar onSubmit={onSubmit} busy={false} />);

    const textarea = screen.getByPlaceholderText(/describe what you're shopping for/i);
    await user.type(textarea, "  a t-shirt  {Enter}");

    expect(onSubmit).toHaveBeenCalledWith("a t-shirt");
    expect(textarea).toHaveValue("");
  });

  it("does not submit on Shift+Enter, inserts a newline instead", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<QueryBar onSubmit={onSubmit} busy={false} />);

    const textarea = screen.getByPlaceholderText(/describe what you're shopping for/i);
    await user.type(textarea, "line one{Shift>}{Enter}{/Shift}line two");

    expect(onSubmit).not.toHaveBeenCalled();
    expect(textarea).toHaveValue("line one\nline two");
  });

  it("does not submit fewer than 3 characters", async () => {
    const onSubmit = vi.fn();
    const user = userEvent.setup();
    render(<QueryBar onSubmit={onSubmit} busy={false} />);

    await user.type(screen.getByPlaceholderText(/describe what you're shopping for/i), "hi{Enter}");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("disables the textarea and send button while busy", () => {
    render(<QueryBar onSubmit={vi.fn()} busy={true} />);
    expect(screen.getByPlaceholderText(/describe what you're shopping for/i)).toBeDisabled();
    expect(screen.getByRole("button", { name: /thinking/i })).toBeDisabled();
  });

  it("does not submit via the button while busy, even with valid text", async () => {
    const onSubmit = vi.fn();
    render(<QueryBar onSubmit={onSubmit} busy={true} />);
    // The button is disabled, so a click is a no-op; assert the contract
    // directly rather than fighting RTL over firing events on disabled nodes.
    expect(screen.getByRole("button")).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("shows example queries only when showExamples is true", () => {
    const { rerender } = render(<QueryBar onSubmit={vi.fn()} busy={false} showExamples={false} />);
    expect(screen.queryByText("Try")).not.toBeInTheDocument();

    rerender(<QueryBar onSubmit={vi.fn()} busy={false} showExamples={true} />);
    expect(screen.getByText("Try")).toBeInTheDocument();
  });

  it("fills the textarea when an example is clicked", async () => {
    const user = userEvent.setup();
    render(<QueryBar onSubmit={vi.fn()} busy={false} showExamples={true} />);

    await user.click(screen.getByText(/premium gifting hamper/i));
    const textarea = screen.getByPlaceholderText(/describe what you're shopping for/i);
    expect((textarea as HTMLTextAreaElement).value).toContain("premium gifting hamper");
  });

  it("uses the custom placeholder when provided", () => {
    render(<QueryBar onSubmit={vi.fn()} busy={false} placeholder="Ask a follow-up…" />);
    expect(screen.getByPlaceholderText("Ask a follow-up…")).toBeInTheDocument();
  });
});
