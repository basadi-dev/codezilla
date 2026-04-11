package ui

import "github.com/charmbracelet/lipgloss"

// ThemeRegistry maps names to available themes
var ThemeRegistry = map[string]func() Theme{
	"tokyonight": ThemeTokyoNight,
	"dracula":    ThemeDracula,
	"catppuccin": ThemeCatppuccin,
}

// AvailableThemes returns a sorted list of available themes
func AvailableThemes() []string {
	return []string{"tokyonight", "dracula", "catppuccin"}
}

// ThemeTokyoNight provides an adaptive theme inspired by TokyoNight
func ThemeTokyoNight() Theme {
	return Theme{
		StyleReset:  lipgloss.NewStyle(),
		StyleRed:    lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#F7768E", Dark: "#F7768E"}),
		StyleGreen:  lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#9ECE6A", Dark: "#9ECE6A"}),
		StyleYellow: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#E0AF68", Dark: "#E0AF68"}),
		StyleBlue:   lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#2E3C64", Dark: "#7AA2F7"}),
		StylePurple: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#BB9AF7", Dark: "#BB9AF7"}),
		StyleCyan:   lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#7DCFFF", Dark: "#7DCFFF"}),
		StyleBold:   lipgloss.NewStyle().Bold(true),
		StyleDim:    lipgloss.NewStyle().Faint(true),

		IconSuccess: "✓",
		IconError:   "✗",
		IconWarning: "⚠",
		IconInfo:    "ℹ",
		IconPrompt:  ">",

		ACTheme: AutocompleteTheme{
			Cmd:       lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#2E3C64", Dark: "#7AA2F7"}),
			Desc:      lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#565F89", Dark: "#565F89"}).Italic(true),
			HiCmd:     lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#5A4A78", Dark: "#BB9AF7"}).Bold(true),
			HiDesc:    lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#4F6832", Dark: "#9ECE6A"}),
			HiPrefix:  lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#8F5E15", Dark: "#E0AF68"}).Bold(true),
			Separator: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#D5D6DB", Dark: "#1F2335"}),
		},
	}
}

// ThemeDracula provides an adaptive theme inspired by Dracula
func ThemeDracula() Theme {
	return Theme{
		StyleReset:  lipgloss.NewStyle(),
		StyleRed:    lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#FF5555", Dark: "#FF5555"}),
		StyleGreen:  lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#50FA7B", Dark: "#50FA7B"}),
		StyleYellow: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#F1FA8C", Dark: "#F1FA8C"}),
		StyleBlue:   lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#6272A4", Dark: "#8BE9FD"}),
		StylePurple: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#BD93F9", Dark: "#BD93F9"}),
		StyleCyan:   lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#8BE9FD", Dark: "#8BE9FD"}),
		StyleBold:   lipgloss.NewStyle().Bold(true),
		StyleDim:    lipgloss.NewStyle().Faint(true),

		IconSuccess: "✓",
		IconError:   "✗",
		IconWarning: "⚠",
		IconInfo:    "ℹ",
		IconPrompt:  ">",

		ACTheme: AutocompleteTheme{
			Cmd:       lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#6272A4", Dark: "#8BE9FD"}),
			Desc:      lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#6272A4", Dark: "#6272A4"}).Italic(true),
			HiCmd:     lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#BD93F9", Dark: "#BD93F9"}).Bold(true),
			HiDesc:    lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#50FA7B", Dark: "#50FA7B"}),
			HiPrefix:  lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#FF79C6", Dark: "#FF79C6"}).Bold(true),
			Separator: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#BFBFBF", Dark: "#44475A"}),
		},
	}
}

// ThemeCatppuccin provides an adaptive theme inspired by Catppuccin (Macchiato/Latte)
func ThemeCatppuccin() Theme {
	return Theme{
		StyleReset:  lipgloss.NewStyle(),
		StyleRed:    lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#D20F39", Dark: "#ED8796"}),
		StyleGreen:  lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#40A02B", Dark: "#A6DA95"}),
		StyleYellow: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#DF8E1D", Dark: "#EED49F"}),
		StyleBlue:   lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#1E66F5", Dark: "#8AADF4"}),
		StylePurple: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#8839EF", Dark: "#C6A0F6"}),
		StyleCyan:   lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#179299", Dark: "#8BD5CA"}),
		StyleBold:   lipgloss.NewStyle().Bold(true),
		StyleDim:    lipgloss.NewStyle().Faint(true),

		IconSuccess: "✓",
		IconError:   "✗",
		IconWarning: "⚠",
		IconInfo:    "ℹ",
		IconPrompt:  ">",

		ACTheme: AutocompleteTheme{
			Cmd:       lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#1E66F5", Dark: "#8AADF4"}),
			Desc:      lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#7C7F93", Dark: "#5B6078"}).Italic(true),
			HiCmd:     lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#8839EF", Dark: "#C6A0F6"}).Bold(true),
			HiDesc:    lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#40A02B", Dark: "#A6DA95"}),
			HiPrefix:  lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#FE640B", Dark: "#F5A97F"}).Bold(true),
			Separator: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#BCC0CC", Dark: "#363A4F"}),
		},
	}
}
