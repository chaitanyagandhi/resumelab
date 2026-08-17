/* The review UI's front end.
 *
 * No framework and no build step: the page is small, and a bundler would put a
 * toolchain between a one-line change and seeing it. Modules are loaded natively.
 */

const statusDot = document.querySelector('[data-role="status-dot"]');
const statusText = document.querySelector('[data-role="status-text"]');

/** Ask the server whether it is there, and say so in the status bar. */
export async function reportHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const health = await response.json();
    setStatus("ok", `ResumeLab ${health.version}`);
    return health;
  } catch (error) {
    // A dead server is the likeliest reason this page is open and doing nothing,
    // so it is worth saying plainly rather than failing silently in the console.
    setStatus("lost", `Not connected: ${error.message}`);
    return null;
  }
}

function setStatus(state, text) {
  statusDot.dataset.state = state;
  statusText.textContent = text;
}

reportHealth();
