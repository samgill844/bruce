project = 'BRUCE'
author = 'Samuel Gill'
copyright = '2026, Samuel Gill'
release = '1.0.0'
extensions = []
master_doc = 'index'
exclude_patterns = ['_build', 'README.md']
html_theme = 'alabaster'
html_theme_options = {'description': 'Exoplanet modelling and photometric analysis',
                      'fixed_sidebar': True, 'sidebar_width': '245px'}
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = 'BRUCE documentation'
html_sidebars = {'**': ['about.html', 'navigation.html', 'searchbox.html']}
