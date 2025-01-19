def render_text(surface, font, text, position, color=(255, 255, 255)):
    """Render text on the surface and returns the new y position."""
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)
    return position[1] + text_surface.get_height()
