import { existsSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const cwd = process.cwd();
const candidates = [".next"];

for (const entry of readdirSync(cwd)) {
  if (entry.startsWith(".next_old_")) {
    candidates.push(entry);
  }
}

for (const name of candidates) {
  const target = join(cwd, name);
  if (existsSync(target)) {
    try {
      rmSync(target, { recursive: true, force: true });
    } catch {
      // ignore EBUSY and similar errors in CI environments
    }
  }
}
