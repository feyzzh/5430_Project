import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import StockHMMViewer from "./StockHMMViewer";

function App() {
  const [count, setCount] = useState(0)
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetch('/api/time').then(res => res.json()).then(data => {
      setCurrentTime(data.time);
    });
  }, []);

  return (
    <>
      {/* <p>The current time is {new Date(currentTime * 1000).toLocaleString()}.</p> */}

      <div>
        <StockHMMViewer />
      </div>
    </>
  )
}

export default App
