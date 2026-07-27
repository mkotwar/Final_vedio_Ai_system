import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import VehicleSearchPage from './VehicleSearchPage'
import Sidebar from '../components/layout/Sidebar'
import { renderWithProviders } from '../test/renderWithProviders'

describe('Vehicle search page', () => {
  it('renders the vehicle search page', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944',
      path: '/search',
    })
    expect(await screen.findByText('Vehicle Search')).toBeInTheDocument()
    expect(await screen.findByText('GVO:RUN_20260725_131944:FA3FCF9E3ABC')).toBeInTheDocument()
  })

  it('includes an All runs option and does not force a default run selection', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search',
      path: '/search',
    })

    expect(await screen.findByText('Vehicle Search')).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'All runs' })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Processing run' })).toHaveValue('')
  })

  it('renders the natural-language field', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944',
      path: '/search',
    })
    expect(await screen.findByPlaceholderText('Find the grey car with plate ending in 6268.')).toBeInTheDocument()
  })

  it('includes Vehicle Search in navigation', () => {
    renderWithProviders(<Sidebar />, { route: '/search', path: '/search' })
    expect(screen.getByText('Vehicle Search')).toBeInTheDocument()
  })

  it('submits filters through the query string', async () => {
    const user = userEvent.setup()
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944',
      path: '/search',
    })

    const plateInput = await screen.findByPlaceholderText('DL8CBF6268')
    await user.clear(plateInput)
    await user.type(plateInput, 'DL8CBF6268')
    await user.click(screen.getByRole('button', { name: 'Search' }))

    await waitFor(() => {
      const calls = (globalThis.fetch as unknown as { mock: { calls: Array<[string]> } }).mock.calls
      expect(calls.some(([url]) => url.includes('/search/vehicles') && url.includes('plate=DL8CBF6268'))).toBe(true)
    })
  })

  it('preserves URL query parameters in the form', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&plate=6268&colour=GREY',
      path: '/search',
    })
    expect(await screen.findByDisplayValue('6268')).toBeInTheDocument()
    expect(await screen.findByDisplayValue('GREY')).toBeInTheDocument()
  })

  it('shows a loading state while searching', () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944',
      path: '/search',
    })
    expect(screen.getByText('Searching vehicles...')).toBeInTheDocument()
  })

  it('submits the natural-language endpoint with selected run and scope', async () => {
    const user = userEvent.setup()
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&result_scope=GLOBAL_VEHICLES',
      path: '/search',
    })

    const input = await screen.findByPlaceholderText('Find the grey car with plate ending in 6268.')
    await user.type(input, 'Find the grey car with plate ending in 6268.')
    await user.click(screen.getByRole('button', { name: 'Search naturally' }))

    await waitFor(() => {
      const calls = (globalThis.fetch as unknown as { mock: { calls: Array<[string, RequestInit | undefined]> } }).mock.calls
      const naturalCall = calls.find(([url, init]) => url.includes('/search/natural-language') && init?.method === 'POST')
      expect(naturalCall).toBeTruthy()
      const payload = JSON.parse(String(naturalCall?.[1]?.body))
      expect(payload.run_code).toBe('RUN_20260725_131944')
      expect(payload.result_scope).toBe('GLOBAL_VEHICLES')
    })
  })

  it('shows interpreted filters and keeps the query visible after natural-language results load', async () => {
    const user = userEvent.setup()
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944',
      path: '/search',
    })

    const input = await screen.findByPlaceholderText('Find the grey car with plate ending in 6268.')
    await user.type(input, 'Find the grey car with plate ending in 6268.')
    await user.click(screen.getByRole('button', { name: 'Search naturally' }))

    expect(await screen.findByText('Interpreted filters')).toBeInTheDocument()
    expect(screen.queryByText('Fallback used')).not.toBeInTheDocument()
    expect(screen.getByDisplayValue('Find the grey car with plate ending in 6268.')).toBeInTheDocument()
    expect(screen.getAllByText('CAM_001, CAM_002').length).toBeGreaterThan(0)
  })

  it('uses the shared wide grouped-card grid for results', async () => {
    const { container } = renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944',
      path: '/search',
    })

    await screen.findByText('GVO:RUN_20260725_131944:FA3FCF9E3ABC')
    expect(container.querySelector('.vehicle-results-grid')).not.toBeNull()
    expect(screen.getAllByText('DL8CBF6268').length).toBeGreaterThan(0)
  })

  it('shows fallback-used badge and empty state for natural-language red-car queries', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&search_mode=natural&natural_query=Show+red+cars.',
      path: '/search',
    })

    expect(await screen.findByText('Fallback used')).toBeInTheDocument()
    expect(await screen.findByText('No results found')).toBeInTheDocument()
  })

  it('renders a clarification message for unknown natural-language cameras', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&search_mode=natural&natural_query=Find+vehicles+in+CAM_999.',
      path: '/search',
    })

    const alert = await screen.findByRole('alert')
    expect(within(alert).getByText('Clarification required')).toBeInTheDocument()
    expect(within(alert).getByText('Camera CAM_999 is not available for the selected run.')).toBeInTheDocument()
  })

  it('applies interpreted filters back into the structured form', async () => {
    const user = userEvent.setup()
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&search_mode=natural&natural_query=Find+the+grey+car+with+plate+ending+in+6268.',
      path: '/search',
    })

    await user.click(await screen.findByRole('button', { name: 'Apply to filters' }))
    expect(await screen.findByDisplayValue('6268')).toBeInTheDocument()
    expect(await screen.findByDisplayValue('GREY')).toBeInTheDocument()
  })

  it('renders provider failures safely', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&search_mode=natural&natural_query=Trigger+provider+failure.',
      path: '/search',
    })

    expect(await screen.findByText('Search unavailable')).toBeInTheDocument()
  })

  it('submits on Enter from the natural-language field', async () => {
    const user = userEvent.setup()
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944',
      path: '/search',
    })

    const input = await screen.findByPlaceholderText('Find the grey car with plate ending in 6268.')
    await user.type(input, 'Find vehicle DL8CBF6268.{enter}')

    await waitFor(() => {
      const calls = (globalThis.fetch as unknown as { mock: { calls: Array<[string, RequestInit | undefined]> } }).mock.calls
      expect(calls.some(([url, init]) => url.includes('/search/natural-language') && init?.method === 'POST')).toBe(true)
    })
  })

  it('shows an empty state for red search results', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&colour=RED',
      path: '/search',
    })
    expect(await screen.findByText('No results found')).toBeInTheDocument()
  })

  it('shows an API error state for invalid class input', async () => {
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&vehicle_class=PLANE',
      path: '/search',
    })
    expect(await screen.findByText('Search unavailable')).toBeInTheDocument()
  })

  it('clears filters back to the default search form', async () => {
    const user = userEvent.setup()
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&plate=6268&colour=GREY&verified_plate_only=true',
      path: '/search',
    })

    await user.click(await screen.findByRole('button', { name: 'Clear filters' }))
    expect((await screen.findByPlaceholderText('DL8CBF6268'))).toHaveValue('')
    expect(screen.getByText('Vehicle Search')).toBeInTheDocument()
  })

  it('requests the correct offset during pagination', async () => {
    const user = userEvent.setup()
    renderWithProviders(<VehicleSearchPage />, {
      route: '/search?run_code=RUN_20260725_131944&limit=1&offset=0',
      path: '/search',
    })

    await user.click(await screen.findByRole('button', { name: 'Next' }))
    await waitFor(() => {
      const calls = (globalThis.fetch as unknown as { mock: { calls: Array<[string]> } }).mock.calls
      expect(calls.some(([url]) => url.includes('/search/vehicles') && url.includes('offset=1'))).toBe(true)
    })
  })
})
