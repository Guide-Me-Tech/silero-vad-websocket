package main

import (
	"fmt"
	"io"
	"log"
	"net/url"
	"os"
	"time"

	"github.com/gorilla/websocket"
)

func TestWebSocketAudioStreaming() {
	// Audio file path
	audioFilePath := "audio.mp3" // Change to your audio file

	// WebSocket server URL
	u := url.URL{Scheme: "ws", Host: "localhost:8080", Path: "/ws"} // Change to your WebSocket endpoint

	// Connect to WebSocket server
	c, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Fatal("dial:", err)
	}
	defer c.Close()

	// Start a goroutine to read messages from the WebSocket
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			_, message, err := c.ReadMessage()
			if err != nil {
				log.Println("read:", err)
				return
			}
			fmt.Printf("Received: %s\n", message)
		}
	}()

	// Open audio file
	file, err := os.Open(audioFilePath)
	if err != nil {
		log.Fatal("Failed to open audio file:", err)
	}
	defer file.Close()

	// Read file in chunks (100ms worth of audio)
	// Note: This is a simplification. In a real implementation,
	// you would need to parse the audio format to determine
	// the actual byte size of 100ms of audio.
	// For example, for 16-bit stereo audio at 44.1kHz:
	// 44100 samples/sec * 2 bytes/sample * 2 channels * 0.1 sec = 17640 bytes per 100ms

	// Assuming 16-bit stereo audio at 16KhZ (adjust as needed)
	chunkSize := 16000 * 2 * 1 / 10 // samples/sec * bytes/sample * channels / (1000ms/100ms)

	buffer := make([]byte, chunkSize)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		// Read a chunk from the file
		n, err := file.Read(buffer)
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Println("read error:", err)
			break
		}

		// Wait for the next tick (to simulate real-time streaming)
		<-ticker.C

		// Send the chunk to the WebSocket
		err = c.WriteMessage(websocket.BinaryMessage, buffer[:n])
		if err != nil {
			log.Println("write:", err)
			return
		}

		fmt.Printf("Sent %d bytes\n", n)
	}

	// Signal that we're done
	c.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))

	// Wait for confirmation that the connection is closed
	select {
	case <-done:
	case <-time.After(time.Second):
	}
}

func main() {
	TestWebSocketAudioStreaming()
}
