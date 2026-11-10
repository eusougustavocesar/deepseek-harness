import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { CONTEXT_WINDOW_EXCEEDED_CODE, createMessage, createUserMessage, LlmAdapter, LlmError } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, LlmResolvedModelInfo, StreamChunk } from '@deepseek-ai/dsh-llm'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { BasicCompactionEngine } from '@deepseek-ai/dsh-compaction-basic'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { Session, SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import { resolveCompactSpec, resolveConfig, resolveTargetPolicy } from '@deepseek-ai/dsh-compaction-basic/src/config.ts'
import { summaryMaxTokens } from '@deepseek-ai/dsh-compaction-basic/src/summarizer.ts'

/** First conversation request overflows; the summary and the retry succeed. */
class LimitedOutputAdapter extends LlmAdapter {
  readonly summaryRequests: GenerateOptions[] = []
  private conversationCalls = 0

  constructor(private readonly maxOutputTokens: number | undefined) {
    super()
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      context: { contextWindow: 128 },
      ...this.maxOutputTokens === undefined ? {} : { maxOutputTokens: this.maxOutputTokens },
    })
  }

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const trailing = options.messages.at(-1)?.content
      .map(block => (block.type === 'text' ? block.text : ''))
      .join('') ?? ''
    if (trailing.includes('acting as a compaction engine')) {
      this.summaryRequests.push(options)
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'block-end', index: 0, block: { type: 'text', text: 'CHECKPOINT' } }
      yield { type: 'finish', reason: { kind: 'stop' } }
      return
    }
    this.conversationCalls += 1
    if (this.conversationCalls === 1) {
      throw new LlmError('request too large for model context', CONTEXT_WINDOW_EXCEEDED_CODE)
    }
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'block-end', index: 0, block: { type: 'text', text: 'recovered' } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

function historySeed(): readonly SessionEvent[] {
  const session = Session.create(SessionId('output-limit-seed'))
  for (let turn = 1; turn <= 2; turn += 1) {
    session.append('turn/start', { turn })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: `history ${turn} ${'old context '.repeat(200)}` }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('step/start', { turn, step: 1 })
    session.append('assistant/message', {
      stream: [],
      turn,
      step: 1,
      message: createMessage({
        role: 'assistant',
        content: [{ type: 'text', text: `answer ${turn} ${'detail '.repeat(200)}` }],
        source: { kind: 'model', provider: 'mock', model: 'mock' },
      }),
    }, { surfaceOp: 'append' })
    session.append('step/end', { turn, step: 1 })
    session.append('turn/end', { turn, reason: { kind: 'completed' } })
  }
  return session.snapshotEvents()
}

async function compactThroughLoop(maxOutputTokens: number | undefined, unknownOutputLimitMaxTokens?: number): Promise<{
  request: GenerateOptions
  recorded: number | undefined
}> {
  const ctx = new Context()
  const adapter = new LimitedOutputAdapter(maxOutputTokens)
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(AgentLoop, { agents: [] })
  await ctx.plugin(TokenMeter)
  ctx.llm.registerAdapter(['mock'], adapter)
  await ctx.plugin(BasicCompactionEngine, {
    headroomTokens: 0,
    thresholdRatio: 1,
    retainTokens: 100,
    maxTokens: 64,
    ...unknownOutputLimitMaxTokens === undefined ? {} : { unknownOutputLimitMaxTokens },
    compactionRetries: 0,
    maxOverflowRetries: 1,
  })
  try {
    const { agent } = await ctx.agentLoop.createAgent(ctx, {
      sessionId: SessionId(`output-limit-${String(maxOutputTokens)}`),
      seed: historySeed(),
      agentOptions: { provider: 'mock', model: 'mock' },
    })
    agent.model.select({ provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'continue' }], source: { kind: 'user' } }))
    await agent.whenIdle()
    expect(adapter.summaryRequests).toHaveLength(1)
    const summaryEvent = agent.session.snapshotEvents().find(event => event.type === 'compaction/summary')
    return {
      request: adapter.summaryRequests[0]!,
      recorded: summaryEvent?.type === 'compaction/summary' ? summaryEvent.data.maxTokens : undefined,
    }
  } finally {
    await ctx.fiber.dispose()
  }
}

describe('summary output limit across the real loop and compaction-basic', () => {
  it('sends and records the model output limit when it is below the configured cap', async () => {
    const { request, recorded } = await compactThroughLoop(40)
    expect(request.maxTokens).toBe(40)
    expect(recorded).toBe(40)
  })

  it('keeps the configured cap when the model limit is larger', async () => {
    const { request, recorded } = await compactThroughLoop(1_000)
    expect(request.maxTokens).toBe(64)
    expect(recorded).toBe(64)
  })

  it('keeps the configured cap when the adapter declares no limit and it is below the fallback', async () => {
    const { request, recorded } = await compactThroughLoop(undefined)
    expect(request.maxTokens).toBe(64)
    expect(recorded).toBe(64)
  })

  it('applies unknownOutputLimitMaxTokens when the adapter declares no limit', async () => {
    const { request, recorded } = await compactThroughLoop(undefined, 48)
    expect(request.maxTokens).toBe(48)
    expect(recorded).toBe(48)
  })
})

describe('summaryMaxTokens', () => {
  function fakeContext(info: Partial<LlmResolvedModelInfo>): { ctx: Context; warn: ReturnType<typeof vi.fn> } {
    const warn = vi.fn()
    const ctx = {
      llm: { resolveModelInfo: () => Promise.resolve({ provider: 'p', id: 'm', name: 'm', ...info }) },
      logger: { warn },
    } as unknown as Context
    return { ctx, warn }
  }

  it('caps the configured value at the declared limit without warning', async () => {
    const { ctx, warn } = fakeContext({ maxOutputTokens: 64_000 })
    const target = { provider: 'anthropic', model: 'claude-opus-4-5' }
    await expect(summaryMaxTokens(ctx, { maxTokens: 65_536, unknownOutputLimitMaxTokens: 32_000 }, target)).resolves.toBe(64_000)
    await expect(summaryMaxTokens(ctx, { maxTokens: 8_192, unknownOutputLimitMaxTokens: 32_000 }, target)).resolves.toBe(8_192)
    expect(warn).not.toHaveBeenCalled()
  })

  it('warns with the route and caps at the fallback when no limit is declared', async () => {
    const { ctx, warn } = fakeContext({})
    const target = { provider: 'gateway', model: 'custom' }
    await expect(summaryMaxTokens(ctx, { maxTokens: 65_536, unknownOutputLimitMaxTokens: 32_000 }, target)).resolves.toBe(32_000)
    await expect(summaryMaxTokens(ctx, { maxTokens: 8_192, unknownOutputLimitMaxTokens: 32_000 }, target)).resolves.toBe(8_192)
    expect(warn).toHaveBeenCalledTimes(2)
    expect(warn.mock.calls[0]![0]).toContain('gateway/custom declares no output limit')
  })

  it('defaults unknownOutputLimitMaxTokens to 32000, honors per-model overrides, and rejects invalid values', () => {
    const target = { provider: 'gateway', model: 'custom' }
    expect(resolveConfig({}).unknownOutputLimitMaxTokens).toBe(32_000)
    expect(resolveTargetPolicy(resolveConfig({
      unknownOutputLimitMaxTokens: 16_000,
      modelPolicies: [{ ...target, unknownOutputLimitMaxTokens: 8_000 }],
    }), target).unknownOutputLimitMaxTokens).toBe(8_000)
    expect(() => resolveConfig({ unknownOutputLimitMaxTokens: 0 })).toThrow(/unknownOutputLimitMaxTokens.*positive integer/)
  })
})

describe('summary fits a near-full Claude Opus 4.5 window under default policy', () => {
  // Claude Opus 4.5: 200,000-token window, 64,000-token output limit.
  const window = 200_000
  const outputLimit = 64_000
  const policy = resolveTargetPolicy(resolveConfig({}), { provider: 'anthropic', model: 'claude-opus-4-5' })

  it.each([0, 32_000, 64_000])('reserved conversation output %i', (reserved) => {
    const spec = resolveCompactSpec(policy, window, reserved)
    const summaryOutput = Math.min(policy.maxTokens, outputLimit)
    // Pressure fires at the threshold; the summary request replays at most that
    // much history, so history plus summary output must fit the window.
    expect(spec.thresholdTokens + summaryOutput).toBeLessThanOrEqual(window)
  })
})
