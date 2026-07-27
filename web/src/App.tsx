import {
  startTransition,
  type CSSProperties,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import {
  Brain,
  Filter,
  Moon,
  Search,
  Sun,
  TerminalSquare,
} from "lucide-react"
import ReactMarkdown from "react-markdown"
import {
  Bar,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
} from "recharts"
import remarkGfm from "remark-gfm"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Input } from "@/components/ui/input"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toggle } from "@/components/ui/toggle"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

type SearchResult = {
  session_id: string
  project: string
  source: string
  role: string
  ts: number
  score?: number | null
  snippet: string
}

type SearchPayload = {
  query: string
  offset: number
  has_more: boolean
  results: SearchResult[]
}

type Message = {
  role: string
  content: string
  ts: number
  tool_name?: string | null
  provisional?: boolean
}

type SessionPayload = {
  session_id: string
  project: string
  source: string
  started_at: number
  ended_at: number
  offset: number
  total: number
  messages: Message[]
}

type PreviewMode = "matches" | "history" | "usage"
type PreviewRow = { message: Message; index: number; context: boolean }

const paramsAtLoad = new URLSearchParams(window.location.search)
const requestedMode = paramsAtLoad.get("mode")
const initialMode: PreviewMode =
  requestedMode === "history" || requestedMode === "usage"
    ? requestedMode
    : "matches"

const formatDate = (timestamp: number) =>
  timestamp
    ? new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(new Date(timestamp))
    : ""

async function api<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: { Accept: "application/json" },
  })
  const data = (await response
    .json()
    .catch(() => ({ error: `HTTP ${response.status}` }))) as T & {
    error?: string
  }
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`)
  return data
}

function getPreferredTheme() {
  const stored = localStorage.getItem("memex-theme")
  if (stored === "dark" || stored === "light") return stored
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
}

type XmlField = { label: string; value: string; path: string }

function parseXml(content: string): { title: string; fields: XmlField[] } | null {
  const source = content.trim()
  if (!/^<[A-Za-z_][\w:.-]*(?:\s[^>]*)?>[\s\S]*>$/.test(source)) return null

  const parser = new DOMParser()
  let documentNode = parser.parseFromString(source, "application/xml")
  let root = documentNode.documentElement
  let fragment = documentNode.querySelector("parsererror") !== null
  if (fragment) {
    documentNode = parser.parseFromString(
      `<memex-fragment>${source}</memex-fragment>`,
      "application/xml",
    )
    if (documentNode.querySelector("parsererror")) return null
    root = documentNode.documentElement
    if (!root.children.length) return null
  }

  const fields: XmlField[] = []
  const walk = (node: Element, parentPath = "") => {
    const path = parentPath ? `${parentPath}/${node.tagName}` : node.tagName
    if (!node.children.length) {
      fields.push({
        label: node.tagName.replace(/[-_]+/g, " "),
        value: node.textContent?.trim() || "",
        path,
      })
      return
    }
    Array.from(node.children).forEach((child) => walk(child, path))
  }

  if (fragment) Array.from(root.children).forEach((child) => walk(child))
  else walk(root)

  return {
    title: fragment
      ? "structured message"
      : root.tagName.replace(/[-_]+/g, " "),
    fields,
  }
}

function XmlMessage({ content }: { content: string }) {
  const parsed = useMemo(() => parseXml(content), [content])
  if (!parsed) return null

  return (
    <div className="xml-card">
      <div className="xml-title">{parsed.title}</div>
      <dl>
        {parsed.fields.map((field, index) => (
          <div className="xml-row" key={`${field.path}-${index}`} title={field.path}>
            <dt>{field.label}</dt>
            <dd>{field.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function MessageContent({ message }: { message: Message }) {
  if (["tool_use", "tool_result"].includes(message.role)) {
    return <pre className="tool-content">{message.content}</pre>
  }

  if (parseXml(message.content)) return <XmlMessage content={message.content} />

  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
    </div>
  )
}

type ActivityMetric = "sessions" | "tokens"

type ActivityPayload = {
  metric: ActivityMetric
  days: number
  token_usage_enabled: boolean
  partial: boolean
  points: Array<{
    date: string
    source: string
    value: number
  }>
}

const activityColors = [
  "oklch(0.55 0.14 255)",
  "oklch(0.62 0.14 155)",
  "oklch(0.66 0.15 65)",
  "oklch(0.58 0.16 320)",
  "oklch(0.62 0.15 20)",
  "oklch(0.58 0.1 205)",
  "oklch(0.5 0.02 260)",
]

const compactNumber = new Intl.NumberFormat(undefined, {
  notation: "compact",
  maximumFractionDigits: 1,
})

function UsageChart({
  active,
  project,
  source,
}: {
  active: boolean
  project: string
  source: string
}) {
  const [metric, setMetric] = useState<ActivityMetric>("sessions")
  const [payload, setPayload] = useState<ActivityPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const requestGeneration = useRef(0)

  useEffect(() => {
    if (!active) return
    const generation = ++requestGeneration.current
    const params = new URLSearchParams({ days: "30", metric })
    if (source !== "all") params.set("source", source)
    if (project.trim()) params.set("project", project.trim())
    setLoading(true)
    setError("")
    void api<ActivityPayload>(`/api/activity?${params}`)
      .then((data) => {
        if (generation === requestGeneration.current) setPayload(data)
      })
      .catch((requestError) => {
        if (generation !== requestGeneration.current) return
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load usage",
        )
      })
      .finally(() => {
        if (generation === requestGeneration.current) setLoading(false)
      })
  }, [active, metric, project, source])

  const { chartConfig, chartData, sources, total } = useMemo(() => {
    const points = payload?.points || []
    const sourceNames = Array.from(new Set(points.map((point) => point.source)))
    const rows = new Map<string, Record<string, string | number>>()
    let sum = 0
    points.forEach((point) => {
      const row = rows.get(point.date) || { date: point.date }
      row[point.source] = Number(row[point.source] || 0) + point.value
      rows.set(point.date, row)
      sum += point.value
    })
    const config = Object.fromEntries(
      sourceNames.map((name) => [name, { label: name }]),
    ) satisfies ChartConfig
    return {
      chartConfig: config,
      chartData: Array.from(rows.values()),
      sources: sourceNames,
      total: sum,
    }
  }, [payload])

  return (
    <div className="usage-surface">
      <div className="usage-heading">
        <div>
          <h2>30-day activity</h2>
          <p>
            {loading
              ? "Loading local activity…"
              : `${compactNumber.format(total)} ${metric}${payload?.partial ? " · partial" : ""}`}
          </p>
        </div>
        <Select
          onValueChange={(value) => setMetric(value as ActivityMetric)}
          value={metric}
        >
          <SelectTrigger aria-label="Usage metric" className="usage-metric">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="sessions">Sessions</SelectItem>
            <SelectItem value="tokens">Tokens</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {error ? (
        <div className="empty text-destructive">{error}</div>
      ) : metric === "tokens" &&
        payload &&
        !payload.token_usage_enabled ? (
        <div className="empty">
          Token usage is disabled. Set <code>token_usage = true</code> in the
          memex config to enable it.
        </div>
      ) : !loading && chartData.length === 0 ? (
        <div className="empty">No activity in this window.</div>
      ) : (
        <ChartContainer className="usage-chart" config={chartConfig}>
          <BarChart accessibilityLayer data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis
              axisLine={false}
              dataKey="date"
              minTickGap={22}
              tickFormatter={(value) =>
                new Intl.DateTimeFormat(undefined, {
                  month: "short",
                  day: "numeric",
                  timeZone: "UTC",
                }).format(new Date(`${value}T00:00:00Z`))
              }
              tickLine={false}
            />
            <YAxis
              axisLine={false}
              tickFormatter={(value) => compactNumber.format(value)}
              tickLine={false}
              width={48}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  formatter={(value, name) => (
                    <>
                      <span className="text-muted-foreground">{name}</span>
                      <span className="ml-auto font-mono font-medium">
                        {compactNumber.format(Number(value))}
                      </span>
                    </>
                  )}
                />
              }
            />
            {sources.map((name, index) => (
              <Bar
                dataKey={name}
                fill={activityColors[index % activityColors.length]}
                key={name}
                radius={index === sources.length - 1 ? [3, 3, 0, 0] : 0}
                stackId="activity"
              />
            ))}
          </BarChart>
        </ChartContainer>
      )}
    </div>
  )
}

function App() {
  const [query, setQuery] = useState(paramsAtLoad.get("q") || "")
  const [source, setSource] = useState(paramsAtLoad.get("source") || "all")
  const [project, setProject] = useState(paramsAtLoad.get("project") || "")
  const [mode, setMode] = useState<PreviewMode>(initialMode)
  const [showThinking, setShowThinking] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [results, setResults] = useState<SearchResult[]>([])
  const [hasMoreResults, setHasMoreResults] = useState(false)
  const [loadingMoreResults, setLoadingMoreResults] = useState(false)
  const [selectedId, setSelectedId] = useState(paramsAtLoad.get("session"))
  const [session, setSession] = useState<SessionPayload | null>(null)
  const [status, setStatus] = useState("Loading recent sessions…")
  const [error, setError] = useState("")
  const [documentCount, setDocumentCount] = useState<number | null>(null)
  const [historyLimit, setHistoryLimit] = useState(150)
  const [theme, setTheme] = useState(getPreferredTheme)
  const searchGeneration = useRef(0)
  const sessionGeneration = useRef(0)
  const sessionCache = useRef(new Map<string, Promise<SessionPayload>>())

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    localStorage.setItem("memex-theme", theme)
  }, [theme])

  const updateLocation = useCallback(
    (nextSelectedId: string | null) => {
      const next = new URLSearchParams()
      if (query.trim()) next.set("q", query.trim())
      if (source !== "all") next.set("source", source)
      if (project.trim()) next.set("project", project.trim())
      if (nextSelectedId) next.set("session", nextSelectedId)
      if (mode !== "matches") next.set("mode", mode)
      history.replaceState({}, "", next.size ? `?${next}` : location.pathname)
    },
    [mode, project, query, source],
  )

  const fetchFirstPage = useCallback((id: string) => {
    const cached = sessionCache.current.get(id)
    if (cached) return cached
    const request = api<SessionPayload>(
      `/api/session?id=${encodeURIComponent(id)}&limit=40`,
    ).catch((requestError) => {
      sessionCache.current.delete(id)
      throw requestError
    })
    sessionCache.current.set(id, request)
    while (sessionCache.current.size > 8) {
      const oldest = sessionCache.current.keys().next().value
      if (oldest) sessionCache.current.delete(oldest)
      else break
    }
    return request
  }, [])

  const searchParamsFor = useCallback(
    (offset: number) => {
      const searchParams = new URLSearchParams({
        limit: "50",
        offset: String(offset),
      })
      if (query.trim()) searchParams.set("q", query.trim())
      if (source !== "all") searchParams.set("source", source)
      if (project.trim()) searchParams.set("project", project.trim())
      return searchParams
    },
    [project, query, source],
  )

  const searchStatus = useCallback(
    (count: number, hasMore: boolean) =>
      count
        ? `${count}${hasMore ? "+" : ""} ${query.trim() ? "matching" : "recent"} session${count === 1 ? "" : "s"}`
        : "No sessions found",
    [query],
  )

  const selectSession = useCallback(
    async (
      id: string,
      summary?: SearchResult,
      shouldUpdateLocation = true,
    ) => {
      setSelectedId(id)
      setHistoryLimit(150)
      setError("")
      if (shouldUpdateLocation) updateLocation(id)

      const generation = ++sessionGeneration.current
      if (summary) {
        setSession({
          session_id: id,
          project: summary.project,
          source: summary.source,
          started_at: summary.ts,
          ended_at: summary.ts,
          offset: 0,
          total: 1,
          messages: [
            {
              role: summary.role,
              content: summary.snippet || "Loading transcript…",
              ts: summary.ts,
              provisional: true,
            },
          ],
        })
      }

      try {
        const firstPage = await fetchFirstPage(id)
        if (generation !== sessionGeneration.current) return
        setSession(firstPage)
        const messages = [...firstPage.messages]
        while (messages.length < firstPage.total) {
          const page = await api<SessionPayload>(
            `/api/session?id=${encodeURIComponent(id)}&offset=${messages.length}&limit=100`,
          )
          if (generation !== sessionGeneration.current) return
          messages.push(...page.messages)
          startTransition(() => setSession({ ...firstPage, messages }))
        }
      } catch (requestError) {
        if (generation !== sessionGeneration.current) return
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load transcript",
        )
      }
    },
    [fetchFirstPage, updateLocation],
  )

  useEffect(() => {
    const timer = window.setTimeout(async () => {
      const generation = ++searchGeneration.current
      const searchParams = searchParamsFor(0)
      setStatus(query.trim() ? "Searching…" : "Loading recent sessions…")
      setError("")
      setHasMoreResults(false)
      setLoadingMoreResults(false)

      try {
        const data = await api<SearchPayload>(`/api/search?${searchParams}`)
        if (generation !== searchGeneration.current) return
        setResults(data.results)
        setHasMoreResults(data.has_more)
        setStatus(searchStatus(data.results.length, data.has_more))

        const currentId = selectedId
        const next =
          data.results.find((item) => item.session_id === currentId) ||
          data.results[0]
        if (next) void selectSession(next.session_id, next, false)
        else {
          setSelectedId(null)
          setSession(null)
        }

      } catch (requestError) {
        if (generation !== searchGeneration.current) return
        const message =
          requestError instanceof Error ? requestError.message : "Search failed"
        setStatus(message)
        setError(message)
      }
    }, 180)
    return () => window.clearTimeout(timer)
  }, [fetchFirstPage, query, searchParamsFor, searchStatus])

  const loadMoreResults = useCallback(async () => {
    if (!hasMoreResults || loadingMoreResults) return
    const generation = searchGeneration.current
    const offset = results.length
    setLoadingMoreResults(true)
    try {
      const data = await api<SearchPayload>(
        `/api/search?${searchParamsFor(offset)}`,
      )
      if (generation !== searchGeneration.current) return
      const known = new Set(results.map((result) => result.session_id))
      const additions = data.results.filter(
        (result) => !known.has(result.session_id),
      )
      const nextCount = results.length + additions.length
      setResults((current) => [...current, ...additions])
      setHasMoreResults(data.has_more)
      setStatus(searchStatus(nextCount, data.has_more))
    } catch (requestError) {
      if (generation !== searchGeneration.current) return
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not load more results",
      )
    } finally {
      if (generation === searchGeneration.current) setLoadingMoreResults(false)
    }
  }, [
    hasMoreResults,
    loadingMoreResults,
    results,
    searchParamsFor,
    searchStatus,
  ])

  useEffect(() => {
    void api<{ documents: number }>("/api/stats")
      .then((data) => setDocumentCount(data.documents))
      .catch(() => {})
  }, [])

  useEffect(
    () => updateLocation(selectedId),
    [mode, selectedId, updateLocation],
  )

  const preview = useMemo(() => {
    if (!session) return { rows: [] as PreviewRow[], noMatches: false, remaining: 0 }
    const visible = session.messages
      .map((message, index) => ({ message, index, context: false }))
      .filter(({ message }) => {
        const tool = ["tool_use", "tool_result", "system"].includes(message.role)
        const thinking = ["reasoning", "thinking"].includes(message.role)
        return (
          (message.provisional || showDetails || !tool) &&
          (showThinking || !thinking)
        )
      })

    if (mode === "history") {
      return {
        rows: visible.slice(0, historyLimit),
        noMatches: false,
        remaining: Math.max(0, visible.length - historyLimit),
      }
    }

    const terms = Array.from(
      new Set(
        query
          .toLocaleLowerCase()
          .split(/\s+/)
          .map((value) =>
            value.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""),
          )
          .filter((value) => value.length >= 2),
      ),
    )
    if (!terms.length)
      return { rows: visible.slice(-12), noMatches: false, remaining: 0 }

    const matches = new Set<number>()
    visible.forEach(({ message, index }) => {
      const text = message.content.toLocaleLowerCase()
      if (terms.some((term) => text.includes(term))) matches.add(index)
    })
    if (!matches.size) return { rows: [], noMatches: true, remaining: 0 }

    const included = new Set<number>()
    matches.forEach((index) => {
      included.add(index - 1)
      included.add(index)
      included.add(index + 1)
    })
    return {
      rows: visible
        .filter(({ index }) => included.has(index))
        .map((row) => ({ ...row, context: !matches.has(row.index) })),
      noMatches: false,
      remaining: 0,
    }
  }, [historyLimit, mode, query, session, showDetails, showThinking])

  const filterCount = Number(source !== "all") + Number(Boolean(project.trim()))
  const transcriptSurface = (
    <div className="transcript-surface">
      <div className="transcript-scroll">
        <div className="messages">
          {error ? (
            <div className="empty text-destructive">{error}</div>
          ) : !session ? (
            <div className="empty">No session to preview.</div>
          ) : preview.noMatches ? (
            <div className="empty">
              This session matched the index, but no literal query terms appear
              in its stored messages.
            </div>
          ) : preview.rows.length === 0 ? (
            <div className="empty">No visible messages in this preview.</div>
          ) : (
            <>
              {preview.rows.map(({ message, index, context }) => (
                <article
                  className={cn("message", context && "context")}
                  key={`${message.ts}-${index}`}
                >
                  <div className="message-meta">
                    <span>{message.tool_name || message.role || "event"}</span>
                    <time>{formatDate(message.ts)}</time>
                  </div>
                  <MessageContent message={message} />
                </article>
              ))}
              {preview.remaining > 0 && (
                <Button
                  className="load-more"
                  onClick={() => setHistoryLimit((limit) => limit + 150)}
                  variant="outline"
                >
                  Show {Math.min(150, preview.remaining)} more
                </Button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )

  return (
    <SidebarProvider
      className="memex-shell"
      style={{ "--sidebar-width": "19rem" } as CSSProperties}
    >
      <Sidebar collapsible="offcanvas" variant="inset">
        <SidebarHeader className="memex-sidebar-header">
          <div className="brand-row">
            <span className="brand-name">memex</span>
            <Badge variant="secondary">local</Badge>
          </div>
          <div className="sidebar-summary">
            <span className={cn(error && "text-destructive")}>{status}</span>
            <span>
              {documentCount === null
                ? "— records"
                : `${documentCount.toLocaleString()} records`}
            </span>
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup className="pt-0">
            <SidebarGroupContent>
              <SidebarMenu>
                {results.map((result) => (
                  <SidebarMenuItem key={result.session_id}>
                    <SidebarMenuButton
                      className="session-button"
                      isActive={selectedId === result.session_id}
                      onClick={() =>
                        void selectSession(result.session_id, result)
                      }
                      onPointerEnter={() =>
                        void fetchFirstPage(result.session_id).catch(() => {})
                      }
                      size="lg"
                      tooltip={result.project || "Untitled session"}
                    >
                      <div className="session-copy">
                        <div className="session-title-row">
                          <strong>
                            {result.project || "Untitled session"}
                          </strong>
                          <time>{formatDate(result.ts)}</time>
                        </div>
                        <div className="session-meta">
                          {result.source} · {result.role}
                          {result.score == null
                            ? ""
                            : ` · ${result.score.toFixed(2)}`}
                        </div>
                        <div className="session-snippet">
                          {result.snippet || "No text preview"}
                        </div>
                      </div>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
                {hasMoreResults && (
                  <SidebarMenuItem>
                    <Button
                      className="load-more-results"
                      disabled={loadingMoreResults}
                      onClick={() => void loadMoreResults()}
                      variant="ghost"
                    >
                      {loadingMoreResults ? "Loading…" : "Load more results"}
                    </Button>
                  </SidebarMenuItem>
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>

      <SidebarInset className="content-inset">
        <Tabs
          className="transcript-tabs"
          onValueChange={(value) => setMode(value as PreviewMode)}
          value={mode}
        >
          <header className="command-bar">
          <SidebarTrigger />
          <InputGroup className="search-group">
            <InputGroupAddon>
              <Search />
            </InputGroupAddon>
            <InputGroupInput
              aria-label="Search conversations"
              autoFocus
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search conversations…"
              value={query}
            />
          </InputGroup>

          <Popover>
            <PopoverTrigger asChild>
              <Button
                aria-label="Filters"
                className="filter-trigger"
                size="icon"
                variant="outline"
              >
                <Filter />
                {filterCount > 0 && (
                  <Badge className="filter-count">{filterCount}</Badge>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="filter-popover">
              <div className="filter-field">
                <label>Source</label>
                <Select onValueChange={setSource} value={source}>
                  <SelectTrigger aria-label="Source" className="w-full">
                    <SelectValue placeholder="All sources" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All sources</SelectItem>
                    <SelectItem value="claude">Claude</SelectItem>
                    <SelectItem value="codex">Codex</SelectItem>
                    <SelectItem value="opencode">OpenCode</SelectItem>
                    <SelectItem value="cursor">Cursor</SelectItem>
                    <SelectItem value="pi">Pi</SelectItem>
                    <SelectItem value="openclaw">OpenClaw</SelectItem>
                    <SelectItem value="copilot">Copilot</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="filter-field">
                <label htmlFor="project-filter">Project</label>
                <Input
                  id="project-filter"
                  onChange={(event) => setProject(event.target.value)}
                  placeholder="Any project"
                  value={project}
                />
              </div>
            </PopoverContent>
          </Popover>

            <TabsList>
              <TabsTrigger value="matches">
                Matches
              </TabsTrigger>
              <TabsTrigger value="history">
                History
              </TabsTrigger>
              <TabsTrigger value="usage">
                Usage
              </TabsTrigger>
            </TabsList>

          <div className="view-toggles">
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex">
                  <Toggle
                    aria-label="Show reasoning"
                    onPressedChange={setShowThinking}
                    pressed={showThinking}
                    variant="outline"
                  >
                    <Brain />
                  </Toggle>
                </span>
              </TooltipTrigger>
              <TooltipContent>Reasoning</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex">
                  <Toggle
                    aria-label="Show tool calls"
                    onPressedChange={setShowDetails}
                    pressed={showDetails}
                    variant="outline"
                  >
                    <TerminalSquare />
                  </Toggle>
                </span>
              </TooltipTrigger>
              <TooltipContent>Tool calls</TooltipContent>
            </Tooltip>
          </div>

          <Button
            aria-label={`Use ${theme === "dark" ? "light" : "dark"} theme`}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            size="icon-sm"
            variant="ghost"
          >
            {theme === "dark" ? <Sun /> : <Moon />}
          </Button>
          </header>

          <TabsContent className="transcript-tab" value="matches">
            {mode === "matches" && transcriptSurface}
          </TabsContent>
          <TabsContent className="transcript-tab" value="history">
            {mode === "history" && transcriptSurface}
          </TabsContent>
          <TabsContent className="transcript-tab" value="usage">
            <UsageChart
              active={mode === "usage"}
              project={project}
              source={source}
            />
          </TabsContent>
        </Tabs>
      </SidebarInset>
    </SidebarProvider>
  )
}

export default App
