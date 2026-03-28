define persistent._show_monikai_buttons = True
define persistent._use_monikai_actions = False
define persistent._use_monikai_ltm = True
define persistent._ltm_destroy_confirm = False

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

        textbutton "Enable Long Term Memory":
            selected persistent._use_monikai_ltm
            action ToggleField(persistent, "_use_monikai_ltm")
            hovered SetField(
                tooltip,
                "value",
                "Enable long term memory storage and retrieval. Monika will remember past conversations."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        textbutton "Force reload memories":
            action Function(renpy.call_in_new_context, "monikai_ltm_reload")
            hovered SetField(
                tooltip,
                "value",
                "Force reload memories from disk. Normally memories from the current session are only visible next session."
            )
            unhovered SetField(tooltip, "value", tooltip.default)

        textbutton "View memory statistics":
            action Function(renpy.call_in_new_context, "monikai_ltm_stats")
            hovered SetField(
                tooltip,
                "value",
                "View current long term memory statistics (memories in RAM, on disk, etc.)."
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