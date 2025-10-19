import { buildServer } from './app'
const port = Number(process.env.PORT || 3000)
buildServer().then(app => {
  app.listen({ port, host: '0.0.0.0' }).then(() => {
    app.log.info({ port }, 'server listening')
  }).catch(err => {
    app.log.error(err, 'failed to listen')
    process.exit(1)
  })
})
