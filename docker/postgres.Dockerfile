FROM postgres:16-alpine@sha256:3c5c8892d184f738f4fe282d14ddaa613a38f00f4189d2d94725ebe6f2909ddb
COPY --chmod=0755 docker/init-databases.sh /docker-entrypoint-initdb.d/01-databases.sh
RUN sed -i 's/\r$//' /docker-entrypoint-initdb.d/01-databases.sh
