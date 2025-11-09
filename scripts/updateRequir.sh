
PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
ENV_DIR="${PROJECT_ROOT}/.cv"

if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "$ENV_DIR" ]; then
    if [ -d "$ENV_DIR" ]; then
        echo "Activating virtual environment: ${ENV_DIR}"
        source "${ENV_DIR}/bin/activate"
    else
        echo "Virtual environment not found: ${ENV_DIR}" >&2
        exit 1
    fi
fi

cd "${PROJECT_ROOT}" || exit 1
uv pip freeze > requirements.txt
echo "Dependencies updated to requirements.txt"