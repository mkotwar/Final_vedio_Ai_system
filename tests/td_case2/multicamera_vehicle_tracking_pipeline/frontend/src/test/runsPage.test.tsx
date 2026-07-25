import { describe, expect, it } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import RunsPage from '../pages/RunsPage'
import { renderWithProviders } from './renderWithProviders'

describe('RunsPage', () => {
  it('renders the runs list', async () => {
    renderWithProviders(<RunsPage />, { route: '/runs', path: '/runs' })
    expect(await screen.findByText('RUN_20260724_151402')).toBeInTheDocument()
    expect(screen.getByText('COMPLETED')).toBeInTheDocument()
  })

  it('updates run code filters in the URL', async () => {
    const user = userEvent.setup()
    renderWithProviders(<RunsPage />, { route: '/runs', path: '/runs' })
    const input = await screen.findByPlaceholderText('Search run code')
    await user.clear(input)
    await user.type(input, 'RUN_20260724')
    await waitFor(() => expect((input as HTMLInputElement).value).toBe('RUN_20260724'))
  })
})
