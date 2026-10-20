from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.container import Container
from app.services.add import AddService
from app.services.search import SearchService
from app.settings import Settings


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings(container: ContainerDep) -> Settings:
    return container.settings


def get_add_service(container: ContainerDep) -> AddService:
    return container.add


def get_search_service(container: ContainerDep) -> SearchService:
    return container.search


SettingsDep = Annotated[Settings, Depends(get_settings)]
AddServiceDep = Annotated[AddService, Depends(get_add_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
