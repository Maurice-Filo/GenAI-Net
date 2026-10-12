import { randomUUID } from "node:crypto";
import { realpathSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";


export const name = "headless-continuing-runner";
export const inject = ["agents", "sessions"];

export function pinnedModelSelection(config) {
  if (typeof config.provider !== "string" || !config.provider.trim() ||
      typeof config.model !== "string" || !config.model.trim()) {
    throw new Error("Campaign runner requires an explicit provider and model; shared defaults are not permitted");
  }
  return { provider: config.provider, model: config.model };
}

export function verifyModelRoutes(events, selection) {
  const routes = events.filter(event => event.type === "request/header")
    .map(event => event.data.header.config);
  if (!routes.length || routes.some(route =>
    route.provider !== selection.provider || route.model !== selection.model)) {
    throw new Error("Campaign model routing mismatch: " + JSON.stringify({ expected: selection, actual: routes }));
  }
  return routes.map(({ provider, model }) => ({ provider, model }));
}

const DEFAULT_MAX_CONTINUATIONS = 2;
const CONTINUATION_PROMPT =
  "Continue the same task from your preceding reasoning. Stop extending the analysis " +
  "and perform the role's required final workspace write now. Do not reread files unless " +
  "strictly necessary. Then return the required final response.";

const internals = {
  stdout: process.stdout,
  stderr: process.stderr,
};

function summarize(events, firstSeq) {
  let started = false;
  let text = "";
  let reason;
  for (const event of events) {
    if (event.seq < firstSeq) continue;
    if (event.type === "turn/start") {
      started = true;
      continue;
    }
    if (!started) continue;
    if (event.type === "assistant/message") {
      const joined = event.data.message.content
        .filter((block) => block.type === "text")
        .map((block) => block.text)
        .join("");
      if (joined !== "") text = joined;
    }
    if (event.type === "turn/end") reason = event.data.reason;
  }
  return { text, reason };
}

async function loadDshApis() {
  const launcher = realpathSync(process.argv[1]);
  const launcherRequire = createRequire(pathToFileURL(launcher));
  const headlessPackage = launcherRequire.resolve(
    "@deepseek-ai/dsh-headless/package.json",
  );
  const dshRequire = createRequire(pathToFileURL(headlessPackage));
  const [agentApi, llmApi, sessionApi] = await Promise.all([
    import(pathToFileURL(dshRequire.resolve("@deepseek-ai/dsh-agent")).href),
    import(pathToFileURL(dshRequire.resolve("@deepseek-ai/dsh-llm")).href),
    import(pathToFileURL(dshRequire.resolve("@deepseek-ai/dsh-session")).href),
  ]);
  return {
    installModelSelection: agentApi.installModelSelection,
    createUserMessage: llmApi.createUserMessage,
    SessionId: sessionApi.SessionId,
  };
}

function fail(io, error) {
  io.stderr.write(`dsh: ${error instanceof Error ? error.message : String(error)}\n`);
  io.exit(1);
}

async function run(ctx, config, io) {
  await ctx.get("loader")?.await();
  const agents = ctx.get("agents");
  const sessions = ctx.get("sessions");
  if (agents === undefined || sessions === undefined) return;

  const { installModelSelection, createUserMessage, SessionId } = await loadDshApis();
  const selection = pinnedModelSelection(config);
  const { agent } = await agents.create({
    sessionId: SessionId(`session-${randomUUID()}`),
    meta: { cwd: process.cwd() },
    agentOptions: {
      provider: selection.provider,
      model: selection.model,
    },
    setup: (agentCtx) => {
      installModelSelection(agentCtx, {
        current: selection,
        assembled: undefined,
      });
    },
  });

  await agent.whenIdle();
  const firstSeq = agent.session.seq;
  const maximum = Number.isInteger(config.maxContinuations)
    ? Math.max(0, config.maxContinuations)
    : DEFAULT_MAX_CONTINUATIONS;
  let task = config.task;
  let continuations = 0;
  let outcome;

  while (true) {
    agent.followup(
      createUserMessage({
        content: [{ type: "text", text: task }],
        source: { kind: "user" },
      }),
    );
    await agent.whenIdle();
    outcome = summarize(agent.session.events, firstSeq);
    if (outcome.reason?.kind !== "max-tokens" || continuations >= maximum) break;
    continuations += 1;
    task = CONTINUATION_PROMPT;
  }

  await sessions.flush(agent.session);
  const actualRoutes = verifyModelRoutes(agent.session.events, selection);
  writeFileSync(`model-route-${agent.session.id}.json`, JSON.stringify({
    expected: selection, actualRoutes,
  }, null, 2) + "\n");
  io.stdout.write(outcome.text + "\n");
  if (outcome.reason?.kind === "error") {
    io.stderr.write(
      `dsh: ${outcome.reason.error.code}: ${outcome.reason.error.message}\n`,
    );
  } else if (outcome.reason?.kind === "max-tokens") {
    io.stderr.write(
      `dsh: model exhausted output tokens after ${continuations} continuation turn(s)\n`,
    );
  }
  io.exit(outcome.reason?.kind === "completed" ? 0 : 1);
}

export function apply(ctx, config) {
  const exit = ctx.get("appExit");
  if (exit === undefined) {
    throw new Error(
      "headless-continuing-runner: the launcher must provide ctx.appExit",
    );
  }
  const io = { stdout: internals.stdout, stderr: internals.stderr, exit };
  run(ctx, config, io).catch((error) => fail(io, error));
}
