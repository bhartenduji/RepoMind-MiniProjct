from collection.repository_manifest import (
    create_repository_record,
    add_repository,
)


REPOSITORIES = [
    {
        "repo_id": "pallets/flask",
        "owner": "pallets",
        "name": "flask",
        "clone_url": "https://github.com/pallets/flask.git",
        "local_path": "data/raw/repos/pallets_flask",
        "language": "python",
        "license_name": "BSD-3-Clause",
        "default_branch": "main",
    },
    {
        "repo_id": "psf/requests",
        "owner": "psf",
        "name": "requests",
        "clone_url": "https://github.com/psf/requests.git",
        "local_path": "data/raw/repos/psf_requests",
        "language": "python",
        "license_name": "Apache-2.0",
        "default_branch": "main",
    },
    {
        "repo_id": "encode/httpx",
        "owner": "encode",
        "name": "httpx",
        "clone_url": "https://github.com/encode/httpx.git",
        "local_path": "data/raw/repos/encode_httpx",
        "language": "python",
        "license_name": "BSD-3-Clause",
        "default_branch": "master",
    },
    {
        "repo_id": "pytest-dev/pytest",
        "owner": "pytest-dev",
        "name": "pytest",
        "clone_url": "https://github.com/pytest-dev/pytest.git",
        "local_path": "data/raw/repos/pytest-dev_pytest",
        "language": "python",
        "license_name": "MIT",
        "default_branch": "main",
    },
    {
        "repo_id": "tiangolo/fastapi",
        "owner": "tiangolo",
        "name": "fastapi",
        "clone_url": "https://github.com/fastapi/fastapi.git",
        "local_path": "data/raw/repos/tiangolo_fastapi",
        "language": "python",
        "license_name": "MIT",
        "default_branch": "master",
    },
    {
        "repo_id": "pydantic/pydantic",
        "owner": "pydantic",
        "name": "pydantic",
        "clone_url": "https://github.com/pydantic/pydantic.git",
        "local_path": "data/raw/repos/pydantic_pydantic",
        "language": "python",
        "license_name": "MIT",
        "default_branch": "main",
    },
    {
        "repo_id": "python-poetry/poetry",
        "owner": "python-poetry",
        "name": "poetry",
        "clone_url": "https://github.com/python-poetry/poetry.git",
        "local_path": "data/raw/repos/python-poetry_poetry",
        "language": "python",
        "license_name": "MIT",
        "default_branch": "main",
    },
    {
        "repo_id": "celery/celery",
        "owner": "celery",
        "name": "celery",
        "clone_url": "https://github.com/celery/celery.git",
        "local_path": "data/raw/repos/celery_celery",
        "language": "python",
        "license_name": "BSD-3-Clause",
        "default_branch": "main",
    },
    {
        "repo_id": "scrapy/scrapy",
        "owner": "scrapy",
        "name": "scrapy",
        "clone_url": "https://github.com/scrapy/scrapy.git",
        "local_path": "data/raw/repos/scrapy_scrapy",
        "language": "python",
        "license_name": "BSD-3-Clause",
        "default_branch": "master",
    },
    {
        "repo_id": "ansible/ansible",
        "owner": "ansible",
        "name": "ansible",
        "clone_url": "https://github.com/ansible/ansible.git",
        "local_path": "data/raw/repos/ansible_ansible",
        "language": "python",
        "license_name": "GPL-3.0",
        "default_branch": "devel",
    },
    {
        "repo_id": "django/django",
        "owner": "django",
        "name": "django",
        "clone_url": "https://github.com/django/django.git",
        "local_path": "data/raw/repos/django_django",
        "language": "python",
        "license_name": "BSD-3-Clause",
        "default_branch": "main",
    },
    {
        "repo_id": "encode/starlette",
        "owner": "encode",
        "name": "starlette",
        "clone_url": "https://github.com/encode/starlette.git",
        "local_path": "data/raw/repos/encode_starlette",
        "language": "python",
        "license_name": "BSD-3-Clause",
        "default_branch": "master",
    },
    {
        "repo_id": "aio-libs/aiohttp",
        "owner": "aio-libs",
        "name": "aiohttp",
        "clone_url": "https://github.com/aio-libs/aiohttp.git",
        "local_path": "data/raw/repos/aio-libs_aiohttp",
        "language": "python",
        "license_name": "Apache-2.0",
        "default_branch": "master",
    },
    {
        "repo_id": "sqlalchemy/sqlalchemy",
        "owner": "sqlalchemy",
        "name": "sqlalchemy",
        "clone_url": "https://github.com/sqlalchemy/sqlalchemy.git",
        "local_path": "data/raw/repos/sqlalchemy_sqlalchemy",
        "language": "python",
        "license_name": "MIT",
        "default_branch": "main",
    },
    {
        "repo_id": "sphinx-doc/sphinx",
        "owner": "sphinx-doc",
        "name": "sphinx",
        "clone_url": "https://github.com/sphinx-doc/sphinx.git",
        "local_path": "data/raw/repos/sphinx-doc_sphinx",
        "language": "python",
        "license_name": "BSD-2-Clause",
        "default_branch": "master",
    },
]


def populate_manifest():
    added = 0
    skipped = 0

    print("\nRepoMind Repository Seed List")
    print("-----------------------------")

    for config in REPOSITORIES:
        record = create_repository_record(
            repo_id=config["repo_id"],
            owner=config["owner"],
            name=config["name"],
            clone_url=config["clone_url"],
            local_path=config["local_path"],
            language=config["language"],
            license_name=config["license_name"],
            default_branch=config["default_branch"],
        )

        record["provenance"] = {
            "source": "curated_seed_list",
            "source_url": config["clone_url"],
        }

        result = add_repository(record)

        if result:
            added += 1
        else:
            skipped += 1

    print("\nSeed Summary")
    print("-----------------------------")
    print(f"Added: {added}")
    print(f"Already existed: {skipped}")


if __name__ == "__main__":
    populate_manifest()