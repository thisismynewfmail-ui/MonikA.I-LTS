define persistent._show_monikai_buttons = True
define persistent._use_monikai_actions = False
define persistent._use_monikai_ltm_injection = True
define persistent._use_monikai_ltm_saving = True

# Setting in Menu to enable/disable the buttons
screen monikai_chat_settings:
    $ tooltip = renpy.get_screen("submods", "screens").scope["tooltip"]

    vbox:
        box_wrap False
        xfill True
        xmaximum 800

        style_prefix "check"

        textbutton "Show buttons":
            selected persistent._show_monikai_buttons
            action ToggleField(persistent, "_show_monikai_buttons")
            hovered SetField(
                tooltip,
                "value",
                "Enable display of shortcut buttons."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        textbutton "Use automatic actions":
            selected persistent._use_monikai_actions
            action ToggleField(persistent, "_use_monikai_actions")
            hovered SetField(
                tooltip,
                "value",
                "Enable Monika to take actions from the chat."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        null height 10

        text "Long Term Memory" style "check_text"

        textbutton "Enable memory injection":
            selected persistent._use_monikai_ltm_injection
            action ToggleField(persistent, "_use_monikai_ltm_injection")
            hovered SetField(
                tooltip,
                "value",
                "Inject relevant past memories into Monika's context. Memories from the current session are NOT used until reloaded or next session."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        textbutton "Enable memory saving":
            selected persistent._use_monikai_ltm_saving
            action ToggleField(persistent, "_use_monikai_ltm_saving")
            hovered SetField(
                tooltip,
                "value",
                "Save new conversations to Monika's long term memory for future sessions."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        textbutton "Force reload memories":
            action Function(renpy.call_in_new_context, "monikai_ltm_reload")
            hovered SetField(
                tooltip,
                "value",
                "Reload memories from disk into RAM. Makes memories saved this session available for injection immediately."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        textbutton "View memory statistics":
            action Function(renpy.call_in_new_context, "monikai_ltm_stats")
            hovered SetField(
                tooltip,
                "value",
                "View current memory statistics: memories loaded in RAM, saved to disk, and last injected."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        textbutton "Destroy all memories":
            action Function(renpy.call_in_new_context, "monikai_ltm_destroy_prompt")
            hovered SetField(
                tooltip,
                "value",
                "WARNING: Permanently delete ALL stored memories. This cannot be undone!"
            )
            unhovered SetField(tooltip, "value", tooltip.default)

# Button for textual chat
screen monika_chatbot_button():
    zorder 15
    style_prefix "hkb"
    vbox:
        xpos 0.05
        yanchor 1.0
        ypos 230
        if renpy.get_screen("hkb_overlay"):
            if store.mas_hotkeys.talk_enabled is False:
                textbutton ("Chatbot"):
                    text_size 20
            else:
                textbutton ("Chatbot"):
                    text_size 20
                    action Jump("monika_chatting_text")

# Button for voice chat
screen monika_voicechat_button():
    zorder 15
    style_prefix "hkb"
    vbox:
        xpos 0.05
        yanchor 1.0
        ypos 280
        if renpy.get_screen("hkb_overlay"):
            if store.mas_hotkeys.talk_enabled is False:
                textbutton ("Voicechat"):
                    text_size 20
            else:
                textbutton ("Voicechat"):
                    text_size 20
                    action Jump("monika_voice_chat")

# Closing the chat
label close_AI:
    show monika idle at t11
    jump ch30_visual_skip
    return
