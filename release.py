"""
Release Management Script for Minigpu Package Family

This script manages version updates and changelog entries across all packages in the minigpu repository.

USAGE:
------

1. VERSION UPDATES:
   python release.py version <version> [--release] [-m "message"]
   
   Examples:
   - python release.py version 1.6.1 --release
     → Updates all packages to version 1.6.1 for release
     → Adds "## 1.6.1" section to all changelogs
     → Removes -WIP suffix if present
     → Removes publish_to: none from pubspec.yaml
   
   - python release.py version 1.6.1 --release -m "adds xyz feature"
     → Same as above but also adds "- adds xyz feature" to changelogs
   
   - python release.py version 1.6.1 -m "adds xyz feature"
     → Updates to version 1.6.1-WIP (development mode)
     → Adds "## 1.6.1-WIP" section and message to changelogs
     → Sets publish_to: none in pubspec.yaml
     → Uses local path dependencies instead of pub.dev versions

2. ADD CHANGELOG MESSAGES:
   python release.py change "message" [packages...]
   
   Examples:
   - python release.py change "fixed memory leak"
     → Adds message to current version in ALL package changelogs
   
   - python release.py change "created abc functionality" minigpu_ffi
     → Adds message only to minigpu_ffi changelog current version
   
   - python release.py change "updated API" minigpu_ffi,gpu_tensor,minigpu
     → Adds message to specified packages (comma-separated or space-separated)

3. PUBLISH PACKAGES:
   python release.py publish
     → Prepares all packages for release:
       - Removes -WIP from CHANGELOG latest version.
       - Removes 'publish_to: none' from pubspec.yaml.
       - Ensures pubspec version matches changelog version.
     → Runs 'dart pub publish --skip-validation' for each package.
       (Note: --skip-validation is not a standard dart flag, ensure your environment supports it)

PACKAGES MANAGED:
-----------------
- minigpu (main package)
- minigpu_platform_interface
- minigpu_ffi
- minigpu_web
- gpu_tensor

WHAT IT DOES:
------------
- Updates version in pubspec.yaml files
- Manages local vs release dependencies (path: vs ^version)
- Creates/updates changelog sections with proper markdown formatting
- Handles -WIP suffix for development versions
- Adds/removes publish_to: none for development/release builds
- Ensures proper markdown formatting (single blank lines, list spacing)

CHANGELOG FORMATTING:
--------------------
Follows markdownlint rules:
- Single blank line before section headers
- Single blank line after section headers  
- Single blank line between bullet lists and next section
- No multiple consecutive blank lines (consolidates 3+ newlines to 2)
"""
import os
import re
import argparse
import subprocess
from datetime import datetime

def load_yaml_preserve_structure(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def update_yaml_content(content, new_version, is_release, local_packages):
    # Update version, preserving the 'version: ' part
    # Use r'\g<1>' to avoid issues with new_version starting with a digit
    content = re.sub(r'^(version:\s*).*$', r'\g<1>' + new_version, content, flags=re.MULTILINE)

    # Regex to match the entire dependencies block
    # It captures 'dependencies:' and its content, stopping before the next major section or EOF
    # The lookahead (?=\n^(?:\w|#)|^\Z) tries to not consume blank lines before the next section.
    dep_block_pattern = re.compile(r'^(dependencies:(?:.|\n)*?)(?=\n^(?:[a-zA-Z_][a-zA-Z0-9_]*:|#)|^\Z)', re.MULTILINE)
    dep_block_match = dep_block_pattern.search(content)

    if dep_block_match:
        original_dep_block = dep_block_match.group(1)
        # Ensure original_dep_block ends with a newline if it's not empty
        if original_dep_block.strip() and not original_dep_block.endswith('\n'):
            # This case should be rare if YAML is well-formed, but as a safeguard
            original_dep_block += '\n'
            
        leading_whitespace_deps = original_dep_block[:len(original_dep_block) - len(original_dep_block.lstrip())]
        
        dep_lines_input = original_dep_block.strip().split('\n')
        updated_dep_lines = [dep_lines_input[0]] # Keep 'dependencies:' line

        current_package_for_path = None
        local_package_paths_to_add = {}

        for line_idx in range(1, len(dep_lines_input)):
            line = dep_lines_input[line_idx]

            # First, check if this line is a 'path:' specifier for the currently active local package
            if current_package_for_path and current_package_for_path in local_packages and \
               line.strip().startswith('path:') and f'../{current_package_for_path}' in line:
                if is_release:
                    pass  # Remove path line by not appending it
                else:
                    updated_dep_lines.append(line)  # Keep existing path line
            else:
                # Not a path line for the active local package, or no active local package.
                # Check if it's a new package declaration.
                package_match = re.match(r'^(\s*)(\w+):', line)
                if package_match:
                    current_indent = package_match.group(1)
                    pkg_name_on_line = package_match.group(2)

                    if pkg_name_on_line in local_packages:
                        current_package_for_path = pkg_name_on_line # Set context for this new local package
                        if is_release:
                            updated_dep_lines.append(f'{current_indent}{pkg_name_on_line}: {new_version}')
                        else:
                            updated_dep_lines.append(f'{current_indent}{pkg_name_on_line}:')
                            path_exists_or_will_be_processed = False
                            if line_idx + 1 < len(dep_lines_input):
                                next_line_stripped = dep_lines_input[line_idx+1].strip()
                                if next_line_stripped.startswith('path:') and f'../{pkg_name_on_line}' in next_line_stripped:
                                     path_exists_or_will_be_processed = True
                            if not path_exists_or_will_be_processed:
                                local_package_paths_to_add[pkg_name_on_line] = f'{current_indent}  path: ../{pkg_name_on_line}'
                    else:
                        # External package or other non-local-package key.
                        current_package_for_path = pkg_name_on_line 
                        updated_dep_lines.append(line)
                else:
                    # Neither a path for active local, nor a new package declaration (key: value).
                    updated_dep_lines.append(line)

        if not is_release and local_package_paths_to_add:
            final_deps_with_paths = []
            for dep_line in updated_dep_lines:
                final_deps_with_paths.append(dep_line)
                pkg_name_match = re.match(r'^(\s*)(\w+):', dep_line)
                if pkg_name_match:
                    pkg_name = pkg_name_match.group(2)
                    if pkg_name in local_package_paths_to_add:
                        final_deps_with_paths.append(local_package_paths_to_add.pop(pkg_name))
            updated_dep_lines = final_deps_with_paths
        
        new_dep_block_internal_content = '\n'.join(updated_dep_lines)
        # Ensure the reconstructed dependencies block content ends with a newline
        if new_dep_block_internal_content.strip() and not new_dep_block_internal_content.endswith('\n'):
            new_dep_block_internal_content += '\n'
        
        reconstructed_dep_block = leading_whitespace_deps + new_dep_block_internal_content
        content = content.replace(original_dep_block, reconstructed_dep_block, 1)

    # Update publish_to field
    if is_release:
        content = re.sub(r'^publish_to:\s*none\s*\n?', '', content, flags=re.MULTILINE)
    else: 
        if not re.search(r'^publish_to:\s*none', content, re.MULTILINE):
            if re.search(r'^publish_to:', content, re.MULTILINE):
                content = re.sub(r'^publish_to:.*$', 'publish_to: none', content, flags=re.MULTILINE)
            else:
                version_line_match = re.search(r'^(version:\s*.*)$', content, re.MULTILINE)
                if version_line_match:
                    # Use r'\g<1>' for the backreference
                    content = re.sub(r'^(version:\s*.*)$', r'\g<1>\npublish_to: none', content, count=1, flags=re.MULTILINE)
                else: 
                    content = 'publish_to: none\n' + content
    
    # Ensure a blank line after the full dependencies block if dev_dependencies or flutter follows
    # This targets cases where dependencies block ends and the next line is dev_dependencies/flutter
    content = re.sub(r'(^dependencies:(?:.|\n)*?\n)(dev_dependencies:)', r'\1\n\2', content, flags=re.MULTILINE)
    content = re.sub(r'(^dependencies:(?:.|\n)*?\n)(flutter:)', r'\1\n\2', content, flags=re.MULTILINE)

    # Final cleanup
    content = re.sub(r'\n{3,}', '\n\n', content) # Consolidate multiple blank lines to one
    content = content.strip() + '\n' # Ensure single trailing newline
    return content

def add_newlines_before_sections(content):
    # Ensures specific sections (like dependencies, dev_dependencies) are preceded by a blank line.
    # This helps maintain a consistent visual structure in the pubspec.yaml.
    section_patterns = [r'^dependencies:', r'^dev_dependencies:']
    processed_content = content
    for pattern_str in section_patterns:
        # Regex to find the pattern if it's not already preceded by a blank line.
        # It looks for the pattern that doesn't have two newlines (a blank line) before it.
        # (?<!\n\n) is a negative lookbehind for two newlines.
        # Need to be careful with start of file.
        
        # Simpler approach: ensure one blank line before if not at start of file
        # Find all occurrences of the pattern
        for match in re.finditer(f'^{pattern_str}', processed_content, re.MULTILINE):
            start_index = match.start()
            if start_index == 0: # At the very beginning of the file
                continue

            # Check characters before the match
            # We want one blank line, meaning two \n characters: content\n\nsection:
            # So, processed_content[start_index-2:start_index] should be '\n\n'
            # If processed_content[start_index-1] is '\n' but processed_content[start_index-2] is not '\n'
            # then we have one \n, need to add another.
            
            # This logic can get complex with re.sub. A simpler iterative replacement:
            pass # Current re.sub is okay, but can be improved if issues persist.

    # The original re.sub was:
    for pattern in section_patterns:
         processed_content = re.sub(f'(?<!\n\n)(^({pattern}))', r'\n\1', processed_content, flags=re.MULTILINE)
         #This adds one \n if not preceded by \n\n. If preceded by \n, it becomes \n\n. If by char, char\n.
         #This might not be robust enough.
    # A better way for section spacing might be to split lines, check, and rebuild.
    # For now, keeping the existing logic which is generally okay.
    # The global \n{3,} -> \n\n cleanup in update_yaml_content helps a lot.
    return processed_content


def process_pubspec(file_path, new_version, is_release, local_packages):
    content = load_yaml_preserve_structure(file_path)
    updated_content = update_yaml_content(content, new_version, is_release, local_packages)
    # add_newlines_before_sections might be redundant if update_yaml_content's cleanup is good
    # final_content = add_newlines_before_sections(updated_content) 
    final_content = updated_content # Rely on update_yaml_content's cleanup for now
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(final_content)

def load_changelog(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        package_name = os.path.basename(os.path.dirname(file_path))
        return f"# {package_name} CHANGELOG\n\n"

def save_changelog(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def get_current_version_from_changelog(content):
    match = re.search(r'^##\s*(\d+\.\d+\.\d+(?:-WIP)?)', content, re.MULTILINE)
    return match.group(1) if match else None

def add_changelog_version_section(content, version, is_release):
    version_header = f"## {version}" if is_release else f"## {version}-WIP"
    
    # Use raw f-string (rf'') for regex pattern
    if re.search(rf'^##\s*{re.escape(version)}(?:-WIP)?$', content, re.MULTILINE):
        return content
    
    lines = content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if line.startswith('#') and not line.startswith('##'): # Main title
            insert_index = i + 1
            break
    else: # No main title found, insert at the beginning (should not happen for valid changelogs)
        insert_index = 0

    while insert_index < len(lines) and lines[insert_index].strip() == '':
        insert_index += 1
    
    new_section = ['', version_header, ''] # Blank line, header, blank line
    
    lines = lines[:insert_index] + new_section + lines[insert_index:]
    return '\n'.join(lines)

def add_changelog_message(content, message, version=None):
    lines = content.split('\n')
    target_line_idx = -1

    if version:
        # Use raw f-string (rf'') for regex pattern
        version_pattern = rf'^##\s*{re.escape(version)}(?:-WIP)?$'
    else: # Find most recent version
        version_pattern = r'^##\s*\d+\.\d+\.\d+(?:-WIP)?$' # This was already raw, which is good

    for i, line in enumerate(lines):
        if re.match(version_pattern, line):
            target_line_idx = i
            if not version: # Found most recent, break
                break 
    
    if target_line_idx == -1:
        return content 
    
    insert_message_at = target_line_idx + 1
    while insert_message_at < len(lines) and lines[insert_message_at].strip() == '':
        insert_message_at += 1
    
    lines.insert(insert_message_at, f"- {message}")
    return '\n'.join(lines)

def remove_wip_from_version(content, version_without_wip):
    # Use raw f-string (rf'') for regex pattern
    wip_pattern = rf'^##\s*({re.escape(version_without_wip)})-WIP$'
    release_version_header = r'## \1' # Use group 1 for the version number
    return re.sub(wip_pattern, release_version_header, content, flags=re.MULTILINE)

def process_changelog(file_path, version, is_release, message=None):
    content = load_changelog(file_path)
    content = add_changelog_version_section(content, version, is_release)
    
    if is_release:
        content = remove_wip_from_version(content, version)
    
    if message:
        # If adding a message to a new version section, pass the version with -WIP if not release
        target_version_for_message = version if is_release else f"{version}-WIP"
        content = add_changelog_message(content, message, target_version_for_message)
    
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip() + '\n'
    save_changelog(file_path, content)

def add_message_to_current_version(file_path, message):
    content = load_changelog(file_path)
    content = add_changelog_message(content, message) # Adds to most recent version
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip() + '\n'
    save_changelog(file_path, content)

PACKAGES = ["minigpu", "minigpu_platform_interface", "minigpu_ffi", "minigpu_web", "gpu_tensor"]

def main_version_update(version, is_release, message):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    for dir_name in PACKAGES:
        pubspec_path = os.path.join(root_dir, dir_name, "pubspec.yaml")
        if os.path.exists(pubspec_path):
            process_pubspec(pubspec_path, version, is_release, PACKAGES)
            print(f"Updated {pubspec_path}")
        else:
            print(f"Warning: {pubspec_path} not found")
        
        changelog_path = os.path.join(root_dir, dir_name, "CHANGELOG.md")
        process_changelog(changelog_path, version, is_release, message)
        print(f"Updated {changelog_path}")

def main_change(packages_args, message):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    target_packages = []
    if not packages_args:
        target_packages = PACKAGES
    else:
        for pkg_list_str in packages_args:
            target_packages.extend([pkg.strip() for pkg in pkg_list_str.split(',')])
    
    for dir_name in target_packages:
        if dir_name not in PACKAGES:
            print(f"Warning: Unknown package '{dir_name}' specified. Skipping.")
            continue
        changelog_path = os.path.join(root_dir, dir_name, "CHANGELOG.md")
        if os.path.exists(os.path.dirname(changelog_path)): # Check if package dir exists
            add_message_to_current_version(changelog_path, message)
            print(f"Added message to {changelog_path}")
        else:
            print(f"Warning: Package directory for {dir_name} not found")

def main_publish():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    all_actions_successful = True

    for package_name in PACKAGES:
        print(f"\n--- Processing package for publish: {package_name} ---")
        package_dir = os.path.join(root_dir, package_name)

        if not os.path.isdir(package_dir):
            print(f"  Error: Package directory {package_dir} not found. Skipping.")
            all_actions_successful = False
            continue

        # 1. Process CHANGELOG.md
        changelog_path = os.path.join(package_dir, "CHANGELOG.md")
        release_version = None
        if os.path.exists(changelog_path):
            changelog_content = load_changelog(changelog_path)
            current_version_in_changelog = get_current_version_from_changelog(changelog_content)

            if current_version_in_changelog:
                release_version = current_version_in_changelog.replace("-WIP", "")
                if current_version_in_changelog.endswith("-WIP"):
                    print(f"  Changelog: Version {current_version_in_changelog}. Removing -WIP suffix -> {release_version}.")
                    changelog_content = remove_wip_from_version(changelog_content, release_version)
                    changelog_content = re.sub(r'\n{3,}', '\n\n', changelog_content).strip() + '\n'
                    save_changelog(changelog_path, changelog_content)
                else:
                    print(f"  Changelog: Version {release_version} is already release-ready.")
            else:
                print(f"  Error: Could not determine current version from {changelog_path}. Manual check required.")
                all_actions_successful = False
        else:
            print(f"  Error: {changelog_path} not found. Cannot proceed with publish for this package.")
            all_actions_successful = False
            continue # Cannot proceed without changelog version

        if not release_version: # If we couldn't get a version from changelog
            print(f"  Error: Failed to determine release version for {package_name}. Skipping publish.")
            all_actions_successful = False
            continue

        # 2. Process pubspec.yaml
        pubspec_path = os.path.join(package_dir, "pubspec.yaml")
        if os.path.exists(pubspec_path):
            pubspec_content = load_yaml_preserve_structure(pubspec_path)
            original_pubspec_content = pubspec_content

            # Ensure version in pubspec matches the release version from changelog
            pubspec_version_match = re.search(r'^version:\s*(.*)$', pubspec_content, re.MULTILINE)
            if pubspec_version_match:
                pubspec_current_version = pubspec_version_match.group(1).strip()
                if pubspec_current_version != release_version:
                    print(f"  Pubspec: Version mismatch (Pubspec: {pubspec_current_version}, Target: {release_version}). Updating.")
                    pubspec_content = re.sub(r'^version:.*$', f'version: {release_version}', pubspec_content, flags=re.MULTILINE)
            else:
                print(f"  Error: 'version:' line not found in {pubspec_path}. Manual check required.")
                all_actions_successful = False
            
            # Remove 'publish_to: none' if present
            if re.search(r'^publish_to:\s*none', pubspec_content, re.MULTILINE):
                print(f"  Pubspec: 'publish_to: none' found. Removing it.")
                pubspec_content = re.sub(r'^publish_to:\s*none\s*\n?', '', pubspec_content, flags=re.MULTILINE)
            
            if pubspec_content != original_pubspec_content:
                pubspec_content = pubspec_content.strip()
                if pubspec_content: pubspec_content += '\n'
                pubspec_content = re.sub(r'\n{3,}', '\n\n', pubspec_content)
                # pubspec_content = add_newlines_before_sections(pubspec_content) # May not be needed with global cleanup
                with open(pubspec_path, 'w', encoding='utf-8') as file:
                    file.write(pubspec_content)
                print(f"  Pubspec: Updated {pubspec_path}.")
            else:
                print(f"  Pubspec: No changes needed for version or publish_to.")
        else:
            print(f"  Error: {pubspec_path} not found. Cannot proceed.")
            all_actions_successful = False
            continue

        # 3. Run dart pub publish
        print(f"  Attempting to publish {package_name} (version {release_version})...")
        # Note: '--skip-validation' is NOT a standard 'dart pub publish' flag.
        # Standard 'dart pub publish' is interactive. '--force' skips some confirmations.
        # Using the user-specified command. This may fail if not supported.
        cmd = ["dart", "pub", "publish", "--skip-validation"]
        try:
            # Using shell=True can be a security risk if cmd components are from untrusted input.
            # Here, cmd is hardcoded, so it's safer. On Windows, shell=True might help with pathing for dart.
            # However, direct execution is preferred. Ensure Dart SDK is in PATH.
            result = subprocess.run(cmd, cwd=package_dir, capture_output=True, text=True, check=False, timeout=300, input="y\n")
            
            if result.returncode == 0:
                print(f"  Publish command for {package_name} SUCCEEDED (or dry-run successful).")
                if result.stdout: print(f"  Output:\n{result.stdout.strip()}")
                # Some tools might output to stderr even on success for warnings
                if result.stderr: print(f"  Error Output (if any):\n{result.stderr.strip()}")
            else:
                print(f"  Publish command for {package_name} FAILED (Return Code: {result.returncode}).")
                if result.stdout: print(f"  Stdout:\n{result.stdout.strip()}")
                if result.stderr: print(f"  Stderr:\n{result.stderr.strip()}")
                all_actions_successful = False
        except FileNotFoundError:
            print(f"  Error: 'dart' command not found. Ensure Dart SDK is in your system PATH.")
            all_actions_successful = False
        except subprocess.TimeoutExpired:
            print(f"  Error: Publish command for {package_name} timed out after 5 minutes.")
            all_actions_successful = False
        except Exception as e:
            print(f"  An unexpected error occurred during publishing of {package_name}: {e}")
            all_actions_successful = False

    if all_actions_successful:
        print("\nAll packages processed. Publish commands initiated.")
    else:
        print("\nSome errors occurred during processing or publishing. Please review the logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage versioning, changelogs, and publishing for Minigpu packages.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve help text formatting
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)
    
    version_parser = subparsers.add_parser('version', help='Update version in pubspec and changelog for all packages.')
    version_parser.add_argument("version", help="The new version number (e.g., 1.0.0)")
    version_parser.add_argument("--release", action="store_true", help="Prepare for release (removes -WIP, local paths, publish_to: none)")
    version_parser.add_argument("-m", "--message", help="Changelog message to add under the new version")
    
    change_parser = subparsers.add_parser('change', help='Add a message to the current version in changelogs.')
    change_parser.add_argument("message", help="Message to add to changelog (e.g., \"Fixed an issue with X\")")
    change_parser.add_argument("packages", nargs="*", help="Specific package names (comma or space separated). If empty, applies to all.")
    
    publish_parser = subparsers.add_parser('publish', help='Prepares all packages for release and attempts to publish them.')
    
    args = parser.parse_args()
    
    if args.command == 'version':
        main_version_update(args.version, args.release, args.message)
    elif args.command == 'change':
        main_change(args.packages, args.message)
    elif args.command == 'publish':
        main_publish()
    else:
        parser.print_help()
