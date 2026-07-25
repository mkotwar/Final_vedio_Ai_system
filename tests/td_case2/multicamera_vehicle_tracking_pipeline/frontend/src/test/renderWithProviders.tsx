import type { ReactElement } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

export function renderWithProviders(
  element: ReactElement,
  options: { route?: string; path?: string } = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[options.route || '/']}>
        <Routes>
          <Route path={options.path || '/'} element={element} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}
