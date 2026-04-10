package main

import (
	"encoding/json"
	"fmt"
	"github.com/charmbracelet/glamour/ansi"
	"github.com/charmbracelet/glamour/styles"
	gl "github.com/charmbracelet/glamour"
)

func main() {
	var s ansi.StyleConfig
	json.Unmarshal([]byte(styles.DarkStyleConfig), &s)
	
	out, _ := json.MarshalIndent(s.Table, "", "  ")
	fmt.Printf("%s\n", out)
}
