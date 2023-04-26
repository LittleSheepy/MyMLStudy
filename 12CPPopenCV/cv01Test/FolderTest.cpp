// FolderTest1
#include <iostream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

void FolderTest1() {
    fs::create_directory("folder");
}
void FolderTest() {
    FolderTest1();
}