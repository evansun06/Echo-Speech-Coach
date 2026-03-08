import { createTheme } from '@mui/material/styles'

const googleSansFont = '"Google Sans", Roboto, sans-serif'

const theme = createTheme({
  typography: {
    fontFamily: 'Roboto, sans-serif',
    h1: { fontFamily: googleSansFont },
    h2: { fontFamily: googleSansFont },
    h3: { fontFamily: googleSansFont },
    h4: { fontFamily: googleSansFont },
    h5: { fontFamily: googleSansFont },
    h6: { fontFamily: googleSansFont },
    subtitle1: { fontFamily: googleSansFont },
    subtitle2: { fontFamily: googleSansFont },
  },
  palette: {
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
    text: {
      primary: '#202124',
      secondary: '#5f6368',
    },
    divider: '#e8eaed',
    primary: {
      main: '#1a73e8',
      dark: '#1967d2',
      contrastText: '#ffffff',
    },
    success: {
      main: '#1e8e3e',
    },
    warning: {
      main: '#f29900',
    },
    error: {
      main: '#d93025',
    },
  },
  shape: {
    borderRadius: 6,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: 'none',
          border: '1px solid #e8eaed',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          textTransform: 'none',
          fontFamily: googleSansFont,
          fontWeight: 500,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontFamily: googleSansFont,
          fontWeight: 500,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 4,
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 6,
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 8,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 6,
          },
        },
      },
    },
  },
})

export default theme
