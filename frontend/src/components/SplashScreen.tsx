import { Box, Fade, Stack, Typography } from '@mui/material'
import { useEffect, useState } from 'react'

function SplashScreen({ onDone }: { onDone: () => void }) {
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const fadeOutTimeoutId = window.setTimeout(() => {
      setVisible(false)
    }, 1600)

    const doneTimeoutId = window.setTimeout(() => {
      onDone()
    }, 2000)

    return () => {
      window.clearTimeout(fadeOutTimeoutId)
      window.clearTimeout(doneTimeoutId)
    }
  }, [onDone])

  return (
    <Fade in={visible} timeout={400}>
      <Box
        sx={{
          minHeight: '100vh',
          bgcolor: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Stack spacing={1.25} alignItems="center">
          <Box
            sx={{
              width: 56,
              height: 56,
              borderRadius: '50%',
              bgcolor: '#1a73e8',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Typography
              sx={{
                fontFamily: '"Google Sans", Roboto, sans-serif',
                fontSize: 22,
                fontWeight: 600,
                lineHeight: 1,
                color: '#ffffff',
              }}
            >
              SC
            </Typography>
          </Box>
          <Typography
            variant="h4"
            sx={{
              fontFamily: '"Google Sans", Roboto, sans-serif',
              fontWeight: 400,
              color: '#202124',
            }}
          >
            Speech Coach
          </Typography>
          <Typography variant="body2" sx={{ color: '#5f6368' }}>
            AI-powered presentation coaching
          </Typography>
        </Stack>
      </Box>
    </Fade>
  )
}

export default SplashScreen
