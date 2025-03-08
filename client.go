// Copyright 2013 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"github.com/streamer45/silero-vad-go/speech"
)

const (
	// Time allowed to write a message to the peer.
	writeWait = 10 * time.Second

	// Time allowed to read the next pong message from the peer.
	pongWait = 60 * time.Second * 5

	// Send pings to peer with this period. Must be less than pongWait.
	pingPeriod = (pongWait * 9) / 10

	// Maximum message size allowed from peer.
	maxMessageSize = 10000 // this should 100 ms audio actually in 16bit, 16000HZ, mono wav or mp3
)

var (
	newline = []byte{'\n'}
	space   = []byte{' '}
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  10000,
	WriteBufferSize: 10000,
}

// Client is a middleman between the websocket connection and the hub.
type Client struct {
	hub *Hub

	// The websocket connection.
	conn *websocket.Conn

	// Buffered channel of outbound messages.
	send chan []byte
}

type VadMessage struct {
	Speech     bool             `json:"speech"`
	TimeStamps []speech.Segment `json:"time_stamps"`
}

func int16ToFloat32(sample int16) float32 {
	return float32(sample) / 32768.0
}

// readPump pumps messages from the websocket connection to the hub.
//
// The application runs readPump in a per-connection goroutine. The application
// ensures that there is at most one reader on a connection by executing all
// reads from this goroutine.
func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()
	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error { c.conn.SetReadDeadline(time.Now().Add(pongWait)); return nil })
	//  load silero vad model
	BytesRead := bytes.NewBuffer(nil)
	sd, err := speech.NewDetector(speech.DetectorConfig{
		ModelPath:            "../../src/silero_vad/data/silero_vad.onnx",
		SampleRate:           16000,
		Threshold:            0.5,
		MinSilenceDurationMs: 100,
		SpeechPadMs:          30,
	})
	if err != nil {
		log.Fatalf("failed to create speech detector: %s", err)
	}

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("error: %v", err)
			}
			break
		}
		BytesRead.Write(bytes.TrimSpace(bytes.Replace(message, newline, space, -1)))
		check_vad := func() VadMessage {
			log.Printf("Bytes Read len: %d", BytesRead.Len())
			numSamples := BytesRead.Len() / 2 // Each sample is 2 bytes (16-bit)
			floatSamples := make([]float32, numSamples)
			for i := 0; i < numSamples; i++ {
				var sample int16
				err := binary.Read(BytesRead, binary.LittleEndian, &sample)
				if err != nil {
					fmt.Println("Error reading sample:", err)
				}
				floatSamples[i] = int16ToFloat32(sample)
			}
			segments, err := sd.Detect(floatSamples)
			if err != nil {
				log.Fatalf("Detect failed: %s", err)
			}

			for _, s := range segments {
				log.Printf("speech starts at %0.2fs", s.SpeechStartAt)
				if s.SpeechEndAt > 0 {
					log.Printf("speech ends at %0.2fs", s.SpeechEndAt)
				}
			}

			// set limits for speech detection
			// limit := 500 // ms

			return VadMessage{false, segments}
		}
		// c.hub.broadcast <- message
		//  check vad for vad for BytesRead
		result := check_vad()

		// marshal to json
		res, err := json.Marshal(&result)
		if err != nil {
			log.Printf("Error while decoding json")
		}
		c.send <- res
	}
	err = sd.Destroy()
	if err != nil {
		log.Fatalf("failed to destroy detector: %s", err)
	}
}

// writePump pumps messages from the hub to the websocket connection.
//
// A goroutine running writePump is started for each connection. The
// application ensures that there is at most one writer to a connection by
// executing all writes from this goroutine.
func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()
	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				// The hub closed the channel.
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued chat messages to the current websocket message.
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// serveWs handles websocket requests from the peer.
func serveWs(hub *Hub, w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println(err)
		return
	}
	client := &Client{hub: hub, conn: conn, send: make(chan []byte, 256)}
	client.hub.register <- client

	// Allow collection of memory referenced by the caller by doing all work in
	// new goroutines.
	go client.writePump()
	go client.readPump()

}
