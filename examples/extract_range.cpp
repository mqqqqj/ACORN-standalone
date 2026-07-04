#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void usage(const char *prog)
{
    std::fprintf(stderr,
                 "Usage: %s --input <file> --output <file> --format <fbin|ibin> \\\n"
                 "       --offset <int> --count <int>\n"
                 "\n"
                 "Extract a contiguous row/id range from an fbin or ibin file.\n"
                 "Offset is zero-based and the extracted range is [offset, offset + count).\n",
                 prog);
    std::exit(1);
}

static void checked_seek(FILE *fp, long long offset, const char *filename)
{
    if (std::fseek(fp, offset, SEEK_SET) != 0)
        throw std::runtime_error(std::string("seek failed: ") + filename);
}

static void extract_fbin(const char *input, const char *output, int offset, int count)
{
    FILE *in = std::fopen(input, "rb");
    if (!in)
        throw std::runtime_error(std::string("cannot open input: ") + input);

    int n = 0, d = 0;
    if (std::fread(&n, sizeof(int), 1, in) != 1 ||
        std::fread(&d, sizeof(int), 1, in) != 1)
    {
        std::fclose(in);
        throw std::runtime_error(std::string("failed to read fbin header: ") + input);
    }
    if (n <= 0 || d <= 0 || offset < 0 || count <= 0 || offset + count > n)
    {
        std::fclose(in);
        throw std::runtime_error("invalid fbin dimensions or requested range");
    }

    std::vector<float> data((size_t)count * d);
    long long byte_offset = 2LL * sizeof(int) + (long long)offset * d * sizeof(float);
    checked_seek(in, byte_offset, input);
    size_t total = (size_t)count * d;
    size_t nread = std::fread(data.data(), sizeof(float), total, in);
    std::fclose(in);
    if (nread != total)
        throw std::runtime_error(std::string("truncated fbin input: ") + input);

    FILE *out = std::fopen(output, "wb");
    if (!out)
        throw std::runtime_error(std::string("cannot open output: ") + output);
    std::fwrite(&count, sizeof(int), 1, out);
    std::fwrite(&d, sizeof(int), 1, out);
    size_t nwritten = std::fwrite(data.data(), sizeof(float), total, out);
    std::fclose(out);
    if (nwritten != total)
        throw std::runtime_error(std::string("truncated fbin output: ") + output);

    std::printf("Extracted fbin rows [%d, %d) from %s to %s (n=%d, d=%d)\n",
                offset, offset + count, input, output, count, d);
}

static void extract_ibin(const char *input, const char *output, int offset, int count)
{
    FILE *in = std::fopen(input, "rb");
    if (!in)
        throw std::runtime_error(std::string("cannot open input: ") + input);

    int n = 0;
    if (std::fread(&n, sizeof(int), 1, in) != 1)
    {
        std::fclose(in);
        throw std::runtime_error(std::string("failed to read ibin header: ") + input);
    }
    if (n <= 0 || offset < 0 || count <= 0 || offset + count > n)
    {
        std::fclose(in);
        throw std::runtime_error("invalid ibin dimensions or requested range");
    }

    std::vector<int> data(count);
    long long byte_offset = sizeof(int) + (long long)offset * sizeof(int);
    checked_seek(in, byte_offset, input);
    size_t nread = std::fread(data.data(), sizeof(int), count, in);
    std::fclose(in);
    if (nread != (size_t)count)
        throw std::runtime_error(std::string("truncated ibin input: ") + input);

    FILE *out = std::fopen(output, "wb");
    if (!out)
        throw std::runtime_error(std::string("cannot open output: ") + output);
    std::fwrite(&count, sizeof(int), 1, out);
    size_t nwritten = std::fwrite(data.data(), sizeof(int), count, out);
    std::fclose(out);
    if (nwritten != (size_t)count)
        throw std::runtime_error(std::string("truncated ibin output: ") + output);

    std::printf("Extracted ibin ids [%d, %d) from %s to %s (n=%d)\n",
                offset, offset + count, input, output, count);
}

int main(int argc, char **argv)
{
    const char *input = nullptr;
    const char *output = nullptr;
    const char *format = nullptr;
    int offset = -1;
    int count = -1;

    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output = argv[++i];
        else if (std::strcmp(argv[i], "--format") == 0 && i + 1 < argc)
            format = argv[++i];
        else if (std::strcmp(argv[i], "--offset") == 0 && i + 1 < argc)
            offset = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc)
            count = std::atoi(argv[++i]);
        else
            usage(argv[0]);
    }

    if (!input || !output || !format || offset < 0 || count <= 0)
        usage(argv[0]);

    try
    {
        if (std::strcmp(format, "fbin") == 0)
            extract_fbin(input, output, offset, count);
        else if (std::strcmp(format, "ibin") == 0)
            extract_ibin(input, output, offset, count);
        else
            usage(argv[0]);
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
