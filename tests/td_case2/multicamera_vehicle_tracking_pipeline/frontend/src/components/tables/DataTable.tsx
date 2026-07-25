import type { ReactNode } from 'react'

export interface DataColumn<Row> {
  key: string
  header: string
  render: (row: Row) => ReactNode
  className?: string
}

interface DataTableProps<Row> {
  columns: Array<DataColumn<Row>>
  rows: Row[]
  getRowKey: (row: Row) => string
  onRowClick?: (row: Row) => void
}

export default function DataTable<Row>({
  columns,
  rows,
  getRowKey,
  onRowClick,
}: DataTableProps<Row>) {
  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} className={column.className} scope="col">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={getRowKey(row)}
              className={onRowClick ? 'data-table__row data-table__row--clickable' : 'data-table__row'}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              tabIndex={onRowClick ? 0 : -1}
              onKeyDown={
                onRowClick
                  ? (event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        onRowClick(row)
                      }
                    }
                  : undefined
              }
            >
              {columns.map((column) => (
                <td key={column.key} className={column.className}>
                  {column.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
