package main

import (
	"encoding/json"
	"fmt"

	"github.com/charmbracelet/glamour/styles"
)

func main() {
	out, _ := json.MarshalIndent(styles.DarkStyleConfig.Table, "", "  ")
	fmt.Printf("%s\n", out)
}
