FROM postgres:16-alpine
COPY --chmod=0755 docker/init-databases.sh /docker-entrypoint-initdb.d/01-databases.sh
