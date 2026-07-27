import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { listRunCameras, listRuns } from '../api/runs'
import { searchVehicles, searchVehiclesNaturalLanguage } from '../api/search'
import VehicleSearchResultCard from '../components/cards/VehicleSearchResultCard'
import EmptyState from '../components/states/EmptyState'
import ErrorState from '../components/states/ErrorState'
import LoadingState from '../components/states/LoadingState'
import Pagination from '../components/tables/Pagination'
import type { CameraListItem } from '../types/run'
import type { InterpretedVehicleSearchFilters, NaturalLanguageSearchRequest, VehicleSearchFilters } from '../types/search'

const VEHICLE_CLASSES = ['3WHEELER', 'BUS', 'CAR', 'MOTORCYCLE', 'TRUCK', 'UNKNOWN']
const VEHICLE_COLOURS = ['BLACK', 'WHITE', 'SILVER', 'GREY', 'RED', 'BLUE', 'GREEN', 'YELLOW', 'ORANGE', 'BROWN', 'BEIGE', 'PURPLE', 'UNKNOWN']

const DEFAULT_FILTERS: VehicleSearchFilters = {
  run_code: '',
  result_scope: 'ALL',
  vehicle_class: '',
  colour: '',
  plate: '',
  plate_match_type: 'CONTAINS',
  camera_codes: '',
  date: '',
  start_time: '',
  end_time: '',
  minimum_confidence: '0.5',
  multi_camera_only: false,
  verified_plate_only: false,
  limit: 25,
  offset: 0,
  sort_by: 'RELEVANCE',
  sort_order: 'DESC',
}

export default function VehicleSearchPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const submittedFilters = useMemo(() => readFilters(searchParams), [searchParams])
  const submittedNaturalQuery = searchParams.get('natural_query') || ''
  const searchMode = searchParams.get('search_mode') || 'structured'
  const [draftFilters, setDraftFilters] = useState<VehicleSearchFilters>(submittedFilters)
  const [naturalQuery, setNaturalQuery] = useState(submittedNaturalQuery)

  useEffect(() => {
    setDraftFilters(submittedFilters)
  }, [submittedFilters])

  useEffect(() => {
    setNaturalQuery(submittedNaturalQuery)
  }, [submittedNaturalQuery])

  const runsQuery = useQuery({
    queryKey: ['search-runs'],
    queryFn: () => listRuns({ page: 1, page_size: 25, status: 'COMPLETED', sort_by: 'created_at', sort_order: 'desc' }),
  })

  const selectedRunCode = draftFilters.run_code || submittedFilters.run_code || ''

  const camerasQuery = useQuery({
    queryKey: ['search-cameras', selectedRunCode],
    queryFn: () => listRunCameras(selectedRunCode, { page: 1, page_size: 100 }),
    enabled: Boolean(selectedRunCode),
  })

  const searchQuery = useQuery({
    queryKey: ['vehicle-search', submittedFilters],
    queryFn: () => searchVehicles(submittedFilters),
    enabled: searchMode !== 'natural',
  })

  const naturalRequest = useMemo<NaturalLanguageSearchRequest | null>(() => {
    if (!submittedNaturalQuery || !submittedFilters.run_code) {
      return null
    }
    return {
      query: submittedNaturalQuery,
      run_code: submittedFilters.run_code || undefined,
      result_scope: submittedFilters.result_scope,
      default_time_tolerance_minutes: 15,
      limit: submittedFilters.limit,
      offset: submittedFilters.offset,
    }
  }, [submittedFilters, submittedNaturalQuery])

  const naturalSearchQuery = useQuery({
    queryKey: ['vehicle-search-natural', naturalRequest],
    queryFn: () => searchVehiclesNaturalLanguage(naturalRequest as NaturalLanguageSearchRequest),
    enabled: searchMode === 'natural' && naturalRequest !== null,
  })

  const selectedCameraCodes = new Set((draftFilters.camera_codes || '').split(',').filter(Boolean))
  const cameraOptions = camerasQuery.data?.items || []

  function updateDraft(name: keyof VehicleSearchFilters, value: string | boolean | number) {
    setDraftFilters((current) => ({
      ...current,
      [name]: value,
    }))
  }

  function submitFilters() {
    const next = new URLSearchParams()
    for (const [key, value] of Object.entries(draftFilters)) {
      if (value === '' || value === false || value === undefined || value === null) {
        continue
      }
      next.set(key, String(value))
    }
    next.set('search_mode', 'structured')
    next.set('offset', '0')
    setSearchParams(next)
  }

  function submitNaturalSearch() {
    const trimmedQuery = naturalQuery.trim()
    if (!trimmedQuery || !selectedRunCode) {
      return
    }
    const next = new URLSearchParams()
    next.set('search_mode', 'natural')
    next.set('natural_query', trimmedQuery)
    next.set('run_code', selectedRunCode)
    next.set('result_scope', draftFilters.result_scope || 'ALL')
    next.set('limit', String(draftFilters.limit || 25))
    next.set('offset', '0')
    next.set('sort_by', draftFilters.sort_by || 'RELEVANCE')
    next.set('sort_order', draftFilters.sort_order || 'DESC')
    setSearchParams(next)
  }

  function clearFilters() {
    const cleared = {
      ...DEFAULT_FILTERS,
      run_code: selectedRunCode,
    }
    setDraftFilters(cleared)
    setNaturalQuery('')
    const next = new URLSearchParams()
    if (selectedRunCode) {
      next.set('run_code', selectedRunCode)
    }
    next.set('search_mode', 'structured')
    next.set('result_scope', 'ALL')
    next.set('limit', '25')
    next.set('offset', '0')
    next.set('sort_by', 'RELEVANCE')
    next.set('sort_order', 'DESC')
    setSearchParams(next)
  }

  function toggleCamera(cameraCode: string) {
    const next = new Set(selectedCameraCodes)
    if (next.has(cameraCode)) {
      next.delete(cameraCode)
    } else {
      next.add(cameraCode)
    }
    updateDraft('camera_codes', Array.from(next).sort().join(','))
  }

  const activeResponse = searchMode === 'natural' ? naturalSearchQuery.data : searchQuery.data
  const activeIsPending = searchMode === 'natural' ? naturalSearchQuery.isPending : searchQuery.isPending
  const activeIsFetching = searchMode === 'natural' ? naturalSearchQuery.isFetching : searchQuery.isFetching
  const activeIsError = searchMode === 'natural' ? naturalSearchQuery.isError : searchQuery.isError
  const refetchActive = () => (searchMode === 'natural' ? naturalSearchQuery.refetch() : searchQuery.refetch())
  const interpretedFilters = searchMode === 'natural' ? naturalSearchQuery.data?.interpreted_filters : null

  function applyInterpretedFilters() {
    if (!interpretedFilters) {
      return
    }
    const structuredFilters = interpretedToDraftFilters(interpretedFilters, selectedRunCode)
    setDraftFilters(structuredFilters)
    const next = new URLSearchParams()
    for (const [key, value] of Object.entries(structuredFilters)) {
      if (value === '' || value === false || value === undefined || value === null) {
        continue
      }
      next.set(key, String(value))
    }
    next.set('search_mode', 'structured')
    next.set('offset', '0')
    setSearchParams(next)
  }

  return (
    <div className="page-stack">
      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Vehicle search</p>
            <h3>Vehicle Search</h3>
            <p className="table-subtext">Natural-language queries are translated into the same validated structured search contract and reuse the existing search service.</p>
          </div>
        </div>
        <div className="search-natural-panel">
          <div className="search-natural-panel__header">
            <div>
              <p className="panel__eyebrow">Search naturally</p>
              <h4>Operator query</h4>
              <p className="table-subtext">Try phrases like “Find the grey car with plate ending in 6268” or “Show cars seen in both cameras”.</p>
            </div>
            <div className="search-natural-panel__actions">
              <button
                className="button button--secondary"
                type="button"
                onClick={submitNaturalSearch}
                disabled={!selectedRunCode || !naturalQuery.trim() || activeIsFetching}
              >
                {activeIsFetching && searchMode === 'natural' ? 'Searching...' : 'Search naturally'}
              </button>
            </div>
          </div>
          <label className="search-natural-panel__input">
            <span>Natural-language search</span>
            <input
              value={naturalQuery}
              onChange={(event) => setNaturalQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  submitNaturalSearch()
                }
              }}
              placeholder="Find the grey car with plate ending in 6268."
            />
          </label>
        </div>
        <div className="filter-bar">
          <label>
            <span>Processing run</span>
            <select value={draftFilters.run_code || ''} onChange={(event) => updateDraft('run_code', event.target.value)}>
              <option value="">All runs</option>
              {runsQuery.data?.items.map((run) => (
                <option key={run.run_code} value={run.run_code}>
                  {run.run_code}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Result scope</span>
            <select value={draftFilters.result_scope || 'ALL'} onChange={(event) => updateDraft('result_scope', event.target.value)}>
              <option value="ALL">All</option>
              <option value="LOCAL_TRACKS">Local tracks</option>
              <option value="GLOBAL_VEHICLES">Global vehicles</option>
            </select>
          </label>
          <label>
            <span>Vehicle class</span>
            <select value={draftFilters.vehicle_class || ''} onChange={(event) => updateDraft('vehicle_class', event.target.value)}>
              <option value="">Any</option>
              {VEHICLE_CLASSES.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Colour</span>
            <select value={draftFilters.colour || ''} onChange={(event) => updateDraft('colour', event.target.value)}>
              <option value="">Any</option>
              {VEHICLE_COLOURS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Plate</span>
            <input value={draftFilters.plate || ''} onChange={(event) => updateDraft('plate', event.target.value)} placeholder="DL8CBF6268" />
          </label>
          <label>
            <span>Plate match type</span>
            <select value={draftFilters.plate_match_type || 'CONTAINS'} onChange={(event) => updateDraft('plate_match_type', event.target.value)}>
              <option value="EXACT">Exact</option>
              <option value="CONTAINS">Contains</option>
              <option value="STARTS_WITH">Starts with</option>
              <option value="ENDS_WITH">Ends with</option>
            </select>
          </label>
          <label>
            <span>Date</span>
            <input type="date" value={draftFilters.date || ''} onChange={(event) => updateDraft('date', event.target.value)} />
          </label>
          <label>
            <span>Start time</span>
            <input type="time" value={draftFilters.start_time || ''} onChange={(event) => updateDraft('start_time', event.target.value)} />
          </label>
          <label>
            <span>End time</span>
            <input type="time" value={draftFilters.end_time || ''} onChange={(event) => updateDraft('end_time', event.target.value)} />
          </label>
          <label>
            <span>Minimum confidence</span>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={draftFilters.minimum_confidence || ''}
              onChange={(event) => updateDraft('minimum_confidence', event.target.value)}
            />
          </label>
          <label>
            <span>Sort by</span>
            <select value={draftFilters.sort_by || 'RELEVANCE'} onChange={(event) => updateDraft('sort_by', event.target.value)}>
              <option value="RELEVANCE">Relevance</option>
              <option value="FIRST_SEEN">First seen</option>
              <option value="LAST_SEEN">Last seen</option>
              <option value="CONFIDENCE">Confidence</option>
              <option value="PLATE">Plate</option>
            </select>
          </label>
          <label>
            <span>Sort order</span>
            <select value={draftFilters.sort_order || 'DESC'} onChange={(event) => updateDraft('sort_order', event.target.value)}>
              <option value="DESC">Descending</option>
              <option value="ASC">Ascending</option>
            </select>
          </label>
        </div>

        <div className="search-checkbox-grid" aria-label="Camera multi-select">
          {cameraOptions.map((camera: CameraListItem) => (
            <label key={camera.camera_code} className="search-checkbox">
              <input
                type="checkbox"
                checked={selectedCameraCodes.has(camera.camera_code || '')}
                onChange={() => toggleCamera(camera.camera_code || '')}
              />
              <span>{camera.camera_code}</span>
            </label>
          ))}
        </div>

        <div className="search-toggle-row">
          <label className="search-toggle">
            <input
              type="checkbox"
              checked={Boolean(draftFilters.multi_camera_only)}
              onChange={(event) => updateDraft('multi_camera_only', event.target.checked)}
            />
            <span>Multi-camera only</span>
          </label>
          <label className="search-toggle">
            <input
              type="checkbox"
              checked={Boolean(draftFilters.verified_plate_only)}
              onChange={(event) => updateDraft('verified_plate_only', event.target.checked)}
            />
            <span>Verified plate only</span>
          </label>
        </div>

        <div className="search-actions">
          <button className="button button--secondary" type="button" onClick={submitFilters} disabled={searchQuery.isFetching || naturalSearchQuery.isFetching}>
            Search
          </button>
          <button className="button button--secondary" type="button" onClick={clearFilters}>
            Clear filters
          </button>
        </div>
      </section>

      {searchMode === 'natural' && naturalSearchQuery.data ? (
        <section className="panel">
          <div className="panel__header">
            <div>
              <p className="panel__eyebrow">Understood as</p>
              <h3>Interpreted filters</h3>
              <p className="table-subtext">{naturalSearchQuery.data.original_query}</p>
            </div>
            <div className="search-result-card__reasons">
              <span className="badge badge--neutral">Provider: {naturalSearchQuery.data.parser.provider}</span>
              {naturalSearchQuery.data.parser.model ? <span className="badge badge--neutral">Model: {naturalSearchQuery.data.parser.model}</span> : null}
              {naturalSearchQuery.data.parser.fallback_used ? <span className="badge badge--warning">Fallback used</span> : null}
            </div>
          </div>
          {naturalSearchQuery.data.clarification_required ? (
            <ErrorState
              title="Clarification required"
              message={naturalSearchQuery.data.clarification_message || 'Please refine the natural-language search request.'}
            />
          ) : null}
          <div className="search-interpretation-grid">
            {renderInterpretationItems(naturalSearchQuery.data.interpreted_filters).map((item) => (
              <div key={item.label} className="metric-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          <div className="search-actions">
            <button className="button button--secondary" type="button" onClick={applyInterpretedFilters}>
              Apply to filters
            </button>
            <button className="button button--secondary" type="button" onClick={submitFilters}>
              Edit filters
            </button>
          </div>
        </section>
      ) : null}

      {activeIsPending ? (
        <LoadingState label={searchMode === 'natural' ? 'Interpreting and searching vehicles...' : 'Searching vehicles...'} />
      ) : activeIsError ? (
        <ErrorState title="Search unavailable" message="The vehicle search request could not be completed." onRetry={() => void refetchActive()} />
      ) : !activeResponse || activeResponse.results.length === 0 ? (
        <EmptyState
          title={searchMode === 'natural' && naturalSearchQuery.data?.clarification_required ? 'Awaiting clarification' : 'No results found'}
          description={
            searchMode === 'natural' && naturalSearchQuery.data?.clarification_required
              ? naturalSearchQuery.data.clarification_message || 'Refine the natural-language request and try again.'
              : submittedFilters.run_code
                ? 'Try broadening the filters for the selected run.'
                : 'Try broadening the filters across all runs.'
          }
        />
      ) : (
        <>
          <section className="panel">
            <div className="panel__header">
              <div>
                <p className="panel__eyebrow">Results</p>
                <h3>{activeResponse.pagination.total} matches</h3>
                <p className="table-subtext">Showing {activeResponse.pagination.returned} results for the current page.</p>
              </div>
            </div>
            <div className="vehicle-results-grid">
              {activeResponse.results.map((result) => (
                <VehicleSearchResultCard
                  key={result.global_vehicle_code || result.track_uuid || `${result.result_type}-${result.first_seen_at}`}
                  result={result}
                />
              ))}
            </div>
          </section>
          <Pagination
            page={Math.floor((activeResponse.pagination.offset || 0) / (activeResponse.pagination.limit || 25)) + 1}
            pageSize={activeResponse.pagination.limit}
            total={activeResponse.pagination.total}
            hasNext={activeResponse.pagination.has_more}
            onPageChange={(page) => {
              const next = new URLSearchParams(searchParams)
              next.set('offset', String((page - 1) * (submittedFilters.limit || 25)))
              setSearchParams(next)
            }}
          />
        </>
      )}
    </div>
  )
}

function renderInterpretationItems(filters: InterpretedVehicleSearchFilters): Array<{ label: string; value: string }> {
  const items: Array<{ label: string; value: string }> = []

  if (filters.run_code) items.push({ label: 'Run', value: filters.run_code })
  if (filters.result_scope) items.push({ label: 'Scope', value: filters.result_scope.replace('_', ' ') })
  if (filters.vehicle_class) items.push({ label: 'Class', value: filters.vehicle_class })
  if (filters.colour) items.push({ label: 'Colour', value: filters.colour })
  if (filters.plate) {
    const matchType = filters.plate_match_type ? filters.plate_match_type.replace('_', ' ').toLowerCase() : 'exact'
    items.push({ label: 'Plate', value: `${matchType} ${filters.plate}` })
  }
  if (filters.camera_codes?.length) items.push({ label: 'Cameras', value: filters.camera_codes.join(', ') })
  if (filters.date) items.push({ label: 'Date', value: filters.date })
  if (filters.start_time) items.push({ label: 'Start time', value: filters.start_time })
  if (filters.end_time) items.push({ label: 'End time', value: filters.end_time })
  if (filters.target_time) items.push({ label: 'Target time', value: filters.target_time })
  if (typeof filters.time_tolerance_minutes === 'number') items.push({ label: 'Tolerance', value: `${filters.time_tolerance_minutes} min` })
  if (typeof filters.minimum_confidence === 'number') items.push({ label: 'Min confidence', value: String(filters.minimum_confidence) })
  if (typeof filters.multi_camera_only === 'boolean') items.push({ label: 'Multi-camera only', value: filters.multi_camera_only ? 'Yes' : 'No' })
  if (typeof filters.verified_plate_only === 'boolean') items.push({ label: 'Verified plate only', value: filters.verified_plate_only ? 'Yes' : 'No' })

  return items
}

function interpretedToDraftFilters(filters: InterpretedVehicleSearchFilters, fallbackRunCode: string): VehicleSearchFilters {
  return {
    ...DEFAULT_FILTERS,
    run_code: filters.run_code || fallbackRunCode,
    result_scope: filters.result_scope || 'ALL',
    vehicle_class: filters.vehicle_class || '',
    colour: filters.colour || '',
    plate: filters.plate || '',
    plate_match_type: filters.plate_match_type || 'CONTAINS',
    camera_codes: filters.camera_codes?.join(',') || '',
    date: filters.date || '',
    start_time: filters.start_time || '',
    end_time: filters.end_time || '',
    minimum_confidence: typeof filters.minimum_confidence === 'number' ? String(filters.minimum_confidence) : '',
    multi_camera_only: Boolean(filters.multi_camera_only),
    verified_plate_only: Boolean(filters.verified_plate_only),
    limit: typeof filters.limit === 'number' ? filters.limit : DEFAULT_FILTERS.limit,
    offset: typeof filters.offset === 'number' ? filters.offset : DEFAULT_FILTERS.offset,
    sort_by: filters.sort_by || DEFAULT_FILTERS.sort_by,
    sort_order: filters.sort_order || DEFAULT_FILTERS.sort_order,
  }
}

function readFilters(searchParams: URLSearchParams): VehicleSearchFilters {
  return {
    ...DEFAULT_FILTERS,
    run_code: searchParams.get('run_code') || '',
    result_scope: (searchParams.get('result_scope') as VehicleSearchFilters['result_scope']) || 'ALL',
    vehicle_class: searchParams.get('vehicle_class') || '',
    colour: searchParams.get('colour') || '',
    plate: searchParams.get('plate') || '',
    plate_match_type: (searchParams.get('plate_match_type') as VehicleSearchFilters['plate_match_type']) || 'CONTAINS',
    camera_codes: searchParams.get('camera_codes') || '',
    date: searchParams.get('date') || '',
    start_time: searchParams.get('start_time') || '',
    end_time: searchParams.get('end_time') || '',
    minimum_confidence: searchParams.get('minimum_confidence') || '0.5',
    multi_camera_only: searchParams.get('multi_camera_only') === 'true',
    verified_plate_only: searchParams.get('verified_plate_only') === 'true',
    limit: Number(searchParams.get('limit') || '25'),
    offset: Number(searchParams.get('offset') || '0'),
    sort_by: (searchParams.get('sort_by') as VehicleSearchFilters['sort_by']) || 'RELEVANCE',
    sort_order: (searchParams.get('sort_order') as VehicleSearchFilters['sort_order']) || 'DESC',
  }
}
