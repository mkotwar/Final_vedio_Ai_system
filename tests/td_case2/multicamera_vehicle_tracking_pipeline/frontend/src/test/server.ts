import { setupServer } from 'msw/node'
import { handlers } from './mockApi'

export const server = setupServer(...handlers)
