import os

def print_directory_tree(path, filepath, prefix="", is_last=True, max_files=2, is_root=False):
    g = open(filepath, 'a', encoding='utf-8')
        # Print the current item (full path if it's the root)
    if is_root:
        g.write(path + '\n')

    g.close()
    # Prefix for child items
    new_prefix = prefix + ("    " if is_last else "│   ")

    if os.path.isdir(path):
        # List all files and directories in the current folder
        items = sorted([os.path.join(path, item) for item in os.listdir(path)])
        dirs = [item for item in items if os.path.isdir(item)]

        for i, item in enumerate(dirs):
            is_last_item = i == len(dirs) - 1

            if item == "...":
                g = open(filepath, 'a', encoding='utf-8')
                g.write(new_prefix + "└── ..." + '\n')
                g.close()
            else:
                # Print the subfolder
                folder_name = os.path.basename(item)
                g = open(filepath, 'a', encoding='utf-8')
                g.write(new_prefix + ("└── " if is_last_item else "├── ") + folder_name + '\n')
                g.close()
                folder_prefix = new_prefix + ("    " if is_last_item else "│   ")

                # For the first two subfolders, show some files
                if i < 3:
                    subdir_items = sorted([os.path.join(item, subitem) for subitem in os.listdir(item)])
                    subdir_files = [f for f in subdir_items if os.path.isfile(f)]

                    # Limit the number of files to show
                    show_files = subdir_files[:max_files]
                    if len(subdir_files) > max_files:
                        show_files.append("...")

                    for j, file in enumerate(show_files):
                        is_last_file = j == len(show_files) - 1
                        g = open(filepath, 'a', encoding='utf-8')
                        if file == "...":
                            g.write(folder_prefix + "└── ..." + '\n')
                        else:
                            g.write(folder_prefix + ("└── " if is_last_file else "├── ") + os.path.basename(file) + '\n')
                        g.close()
            print_directory_tree(item, filepath, new_prefix, is_last_item, max_files)

# Print directory tree
data_path = 'CV Dataset'
print("📂 NYU Depth v2 - Directory Tree:\n")
print_directory_tree(data_path, 'DirTree.txt', is_root=True)