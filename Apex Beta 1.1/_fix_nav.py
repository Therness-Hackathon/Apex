filepath = r'c:\New folder\Apex Beta 1.1\dashboard_app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the 4 nav NavItem lines with tab- IDs
found_indices = []
for i, line in enumerate(lines):
    if 'dbc.NavItem(dbc.NavLink(' in line and any(
        tid in line for tid in ['tab-overview', 'tab-perf', 'tab-sub', 'tab-about']
    ):
        found_indices.append(i)

print('Found nav lines at:', [i+1 for i in found_indices])
for idx in found_indices:
    print(f'  L{idx+1}: {repr(lines[idx][:100])}')

if len(found_indices) == 4:
    new_nav = [
        '            dbc.NavItem(dbc.NavLink("About",       href="#", id="tab-about",   n_clicks=0, active=True)),\n',
        '            dbc.NavItem(dbc.NavLink("Overview",    href="#", id="tab-overview", n_clicks=0)),\n',
        '            dbc.NavItem(dbc.NavLink("Performance", href="#", id="tab-perf",    n_clicks=0)),\n',
        '            dbc.NavItem(dbc.NavLink("Submission",  href="#", id="tab-sub",     n_clicks=0)),\n',
    ]
    for i, idx in enumerate(found_indices):
        lines[idx] = new_nav[i]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('Nav updated successfully!')
else:
    print(f'ERROR: Expected 4 nav lines, found {len(found_indices)}')
